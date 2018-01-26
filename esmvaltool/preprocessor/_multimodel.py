"""Functions for multi-model operations
   supports a multitude of multimodel statistics
   computations; the only requisite is the ingested
   cubes have (TIME-LAT-LON) or (TIME-PLEV-LAT-LON)
   dimensions; and obviously consistent units.
"""
import logging
from functools import reduce

import os
from datetime import datetime as dd
from datetime import timedelta as td
import iris
import numpy as np

from ._io import save_cubes

logger = logging.getLogger(__name__)


def _compute_stats(cubes, statname):
    """compute any multimodel stats"""
    if statname == 'means':
        statistic = np.ma.mean([s.data for s in cubes], axis=0)
    elif statname == 'medians':
        statistic = np.ma.median([s.data for s in cubes], axis=0)
    return statistic


def _put_in_cube(cubes, stats_name, ncfiles, fname):
    """quick cube building and saving"""
    # grab coordinates from any cube
    times = cubes[0].coord('time')
    lats = cubes[0].coord('latitude')
    lons = cubes[0].coord('longitude')
    if len(cubes[0].shape) == 3:
        cspec = [(times, 0), (lats, 1), (lons, 2)]
    elif len(cubes[0].shape) == 4:
        plev = cubes[0].coord('air_pressure')
        cspec = [(times, 0), (plev, 1), (lats, 2), (lons, 3)]
    # compute stats and put in cube
    dspec = _compute_stats(cubes, stats_name)
    fixed_dspec = np.ma.fix_invalid(dspec, copy=False, fill_value=1e+20)
    stats_cube = iris.cube.Cube(fixed_dspec,
                                dim_coords_and_dims=cspec,
                                long_name=stats_name)
    coord_names = [coord.name() for coord in cubes[0].coords()]
    if 'air_pressure' in coord_names:
        if len(cubes[0].shape) == 3:
            stats_cube.add_aux_coord(cubes[0].coord('air_pressure'))
    stats_cube.attributes['_filename'] = fname
    stats_cube.attributes['NCfiles'] = str(ncfiles)
    return stats_cube


def _sdat(srl_no, unit_type):
    """convert to a datatime point"""
    if unit_type == 'day since 1950-01-01 00:00:00.0000000':
        new_date = dd(1950, 1, 1, 0) + td(srl_no)
    elif unit_type == 'day since 1850-01-01 00:00:00.0000000':
        new_date = dd(1850, 1, 1, 0) + td(srl_no)
    # add more supported units here
    return new_date


def _get_overlap(cubes):
    """Get discrete time overlaps."""
    utype = str(cubes[0].coord('time').units)
    all_times = [cube.coord('time').points for cube in cubes]
    bounds = [range(int(b[0]), int(b[-1]) + 1) for b in all_times]
    time_pts = reduce(np.intersect1d, (i for i in bounds))
    if len(time_pts) > 1:
        return _sdat(time_pts[0], utype), _sdat(time_pts[-1], utype)


def _slice_cube(cube, min_t, max_t):
    """slice cube on time"""
    fmt = '%Y-%m-%d-%H'
    ctr = iris.Constraint(time=lambda x:
                          min_t <= dd.strptime(x.point.strftime(fmt),
                                               fmt) <= max_t)
    cube_slice = cube.extract(ctr)
    return cube_slice


def multi_model_mean(cubes, span, filename, exclude):
    """Compute multi-model mean and median."""
    logger.debug('Multi model statistics: excluding files: %s', str(exclude))
    selection = [
        cube for cube in cubes
        if not all(cube.attributes.get(k) in exclude[k] for k in exclude)
    ]

    if len(selection) < 2:
        logger.info("Single model in list: will not compute statistics.")
        return cubes

    # check if we have any time overlap
    if _get_overlap(cubes) is None:
        logger.info("Time overlap between cubes is none or a single point.")
        logger.info("check models: will not compute statistics.")
        return cubes
    else:
        # add file name info
        file_names = [
            os.path.basename(cube.attributes.get('_filename'))
            for cube in cubes
        ]

        # look at overlap type
        if span == 'overlap':
            tmeans = []
            tx1, tx2 = _get_overlap(cubes)
            for cube in selection:
                logger.debug("Using common time overlap between "
                             "models to compute statistics.")
                logger.debug("Bounds: %s and %s", str(tx1), str(tx2))
                # slice cube on time
                with iris.FUTURE.context(cell_datetime_objects=True):
                    cube = _slice_cube(cube, tx1, tx2)

                # record data
                tmeans.append(cube)

        elif span == 'full':
            logger.debug("Using full time spans "
                         "to compute statistics.")
            tmeans = []
            # lay down time points
            tpts = [c.coord('time').points for c in selection]
            tx = list(set().union(*tpts))
            tx.sort()

            # loop through cubes and apply masks
            for cube in selection:
                # construct new shape
                fine_shape = tuple([len(tx)] + list(cube.data.shape[1:]))
                # find indices of present time points
                oidx = [tx.index(s) for s in cube.coord('time').points]
                # reshape data to include all possible times
                ndat = np.ma.resize(cube.data, fine_shape)

                # build the time mask
                c = np.ones(fine_shape, bool)
                for ti in oidx:
                    c[ti] = False
                ndat.mask |= c

                # build the new coords
                # preserve units
                time_c = iris.coords.DimCoord(
                    tx, standard_name='time',
                    units=cube.coord('time').units)
                lat_c = cube.coord('latitude')
                lon_c = cube.coord('longitude')

                # time-lat-lon
                if len(fine_shape) == 3:
                    cspec = [(time_c, 0), (lat_c, 1), (lon_c, 2)]
                # time-plev-lat-lon
                elif len(fine_shape) == 4:
                    pl_c = cube.coord('air_pressure')
                    cspec = [(time_c, 0), (pl_c, 1), (lat_c, 2), (lon_c, 3)]

                # build cube
                ncube = iris.cube.Cube(ndat, dim_coords_and_dims=cspec)
                coord_names = [coord.name() for coord in cube.coords()]
                if 'air_pressure' in coord_names:
                    if len(fine_shape) == 3:
                        ncube.add_aux_coord(cube.coord('air_pressure'))
                ncube.attributes = cube.attributes
                tmeans.append(ncube)

        else:
            logger.debug("No type of time overlap specified "
                         "- will not compute cubes statistics")
            return cubes

    c_mean = _put_in_cube(tmeans, 'means', file_names, filename)

    c_med = _put_in_cube(tmeans, 'medians', file_names, filename)

    save_cubes([c_mean, c_med])

    return cubes
