import numpy as np
import time
from collections import defaultdict
# TODO:
#   * currently, the annotation format implies that each column can only
#     exist in one plot, that might not be the case. Can be worked around
#     by creating a copy of the column with another name
#   * add axis notation, e.g. "USD", "Date"
#   * disclaimer: Past performance is no guarantee of future performance etc.
#   * draw lines with marker for effect? e.g. big blue dot?
#   * what happens with sleep, if two annotations same day, test it
#   * arrowprops as parameter to annotation, put in the annotation dict?
#   * fmtyaxis below, move outside and pass function as parameter?


def fmt_milnodec(x, pos):
    """Number format with thousandseparator, no decimals, e.g. 10,000

    code example::

    >>>fmtyaxis(fmt_milnodec)
    """
    return format(x, ',.0f')


def fmtyaxis(plt, ax, function):
    """Formats current y-axis with given function

    code example::

    >>>fmtyaxis(fmt_dollars)
    """
    yformat = plt.FuncFormatter(function)
    ax.yaxis.set_major_formatter(yformat)


class Animator():
    """ Animator animates a numpy arrays in one or more subplots, using
    a specified time series and data arrays and the specified type of plot (currently
    support line and scatter plots).

    Annotations can be added. This can be used to e.g. create case studies or
    explainers.

    Note that the animate() function returned by get_animate_function() must
    be run in the correct order, otherwise the behaviour is undefined.

    :param time_series: the Time Series. The time series array determines the timeline
    of the animation.
    :type time_series: numpy array.
    :param data_per_subplot: a list of subplots, with a dictionary of x-values and y-values
    for each subplot. Maximum two y-values per x-value per subplot are currently supported.
    :type data_per_subplot: [{'xvalues': {name as str: numpy array},
                              'yvalues': {name as str: numpy array,
                                          name as str: numpy array},
                              },
                              {'xvalues': {name as str: numpy array},
                              'yvalues': {name as str: numpy array,
                                          name as str: numpy array},
                              })]  

    :param annotations: a nested dict of all the annotations. First key is
    the y-value name of the annotation, second key the index, and the last
    dict contains the meta data with possible keys. For line plots the possible keys
    are point_to_date, text, x_offset, y_offset, scatter_point, arrow,
    sleep_frames, visible_rows. For scatter plots the possible keys are data_point_index_or_data, 
    text, point_xvalue, point_yvalue, x_offset, y_offset, sleep_frames, visible_rows. 
    Note that the index key needs to match exactly the index of the time series 
    (not only be equal to), so if the index is of numpy.datetime64, ensure that the 
    frequency resolution is the same, e.g. 'ns' (nanoseconds).
    :type annotations: {{{}}}
    
    :param horizontal_lines: a nested dict of all the horizontal lines. First key is
    the y-value name of the annotation, second key the index, and the last
    dict contains the meta data with possible keys. For line plots the possible keys
    are xmin, xmax, xtext, text, yval, x_offset, y_offset, sleep_frames, visible_rows. 
    Note that the index key needs to match exactly the index of the time series 
    (not only be equal to), so if the index is of numpy.datetime64, ensure that the 
    frequency resolution is the same, e.g. 'ns' (nanoseconds).
    :type horizontal_lines: {{{}}}
    :param type_of_graphs: list of types of graphs for each subplot. Currently only
    support 'line' and 'scatter'. If not specified defaults to line plots only.
    :type type_of_graphs: [str]
    :param alignment: for more than one subplot can either have subplots aligned
    vertically ('vertical') or horizontally ('horizontal'). By default horizontal.
    :type alignment: str
    :param plt: the matplotlib pyplot object, so it can be styled before sent
    in. This class does not import matplotlib.
    :type plt: matplotlib.pyplot
    :param figsize: the figure size: width, height in inches.
    :type figsize: (int, int)
    :param titles_per_subplot: titles of each subplots as a list of strings
    :type titles_per_subplot: [str]
    :param use_legends: if legends should be used
    :type use_legends: bool
    :param legends_loc: location for legends as per pyplot, default is
    'lower left'
    :type legends_loc: str
    :param ylim_min_multiplier_default: the default multiplier to use for
    the minimum y axis value for plots, e.g. 0 or 0.9 (default)
    :type ylim_min_multiplier_default: int
    :param ylim_max_multiplier_default:  the default multiplier to use for
    the maximum y axis value for plots, e.g. 1.05 (default)
    :type ylim_max_multiplier_default: int
    :param speed: the speed, how many rows to skip each frame
    :type speed: int
    :param attribs: additional parameters used to customise each. The first
    key is the plot number and value is another dict with the following
    possible parameters: y_min_multiplier, y_max_multiplier
    :type attribs: dict
    """
    def __init__(
            self,
            time_series,
            data_per_subplot,
            plt,
            axis_limits = None,
            subplot_alignment='horizontal',
            annotations=None,
            horizontal_lines=None,
            figsize=None,
            titles_per_subplot=None,
            use_legends=False,
            legends_loc=None,
            legends_rename=None,
            ylim_min_multiplier_default=None,
            ylim_max_multiplier_default=None,
            speed=1,
            attribs=None):
        self.time_series = time_series
        self.data_per_subplot = data_per_subplot
        self.plt = plt
        self.subplot_alignment = subplot_alignment
        self.axis_limits = axis_limits
        self.annotations_given = annotations
        self.h_lines_given = horizontal_lines
        self.figsize = figsize
        self.titles_per_subplot = titles_per_subplot
        self.use_legends = use_legends
        self.legends_loc = legends_loc
        self.legends_rename = legends_rename
        self.ylim_min_multiplier_default = ylim_min_multiplier_default
        self.ylim_max_multiplier_default = ylim_max_multiplier_default
        self.attribs = attribs
        
        # columns_lhs_per_subplot = list of list of str
        if self.attribs is None:
            self.attribs = {}
        self.DEBUG = True
        self.SPEED = speed
        assert(type(self.SPEED) is int), 'speed needs to be int'
        
        self.YLIM_MIN_MULTIPLIER = ylim_min_multiplier_default or 0.90
        self.YLIM_MAX_MULTIPLIER = ylim_max_multiplier_default or 1.05
        # ^ extend y axis by x multiplier x max

        self.sleep_lag = 0
        # ^ total sleep lag incurred so far
        self.step_size_cumulative = 0
        # ^ total step size speed up cumulated so far
        if legends_loc is None:
            self.legends_loc = 'lower left'
        else:
            self.legends_loc = legends_loc
        nbr_subplots = len(data_per_subplot)
        if annotations is None:
            annotations = {}
        if horizontal_lines is None:
            horizontal_lines = {}
        if axis_limits is not None:
            assert(len(axis_limits) == len(data_per_subplot))
            for j in range(len(axis_limits)):
                assert(len(axis_limits[j]) == 2)
                for k in range(len(axis_limits[j])):
                    assert(len(axis_limits[j][k]) == 2)

        self.annotations_meta = annotations
        self.hlines_meta = horizontal_lines
        self.line_names = []  # name of each line, in same order as line
        self.lines = []  # the line objects, in same order as above
        self.lines_type = [] # the line type. Two types supported: expand_line - getting longer with time, 
                             #  changing_line - at each time different line 
        self.lines_x_values = []  # the x values for each line, same order as ^
        self.lines_y_values = []  # the y values for each line, same order as ^
        self.lines_ax = []  # for each line idx, what is the axes used
        print(len(self.time_series))
        self.scatter_names = []  # name of each scatter plot, in same order as scaters
        self.scatters = []  # the scatters objects, in same order as above
        self.scatters_values_y = []  # the y values for each scatter, same order as ^
        self.scatters_values_x = []  # the x values for each scatter, same order as ^
        self.scatters_ax = []  # for each scatter idx, what is the axes used

        self.contour_names = []  # name of each contour plot, in same order as contours
        self.contours = []  # the contour objects, in same order as above
        self.contours_values_y = []  # the y values for each contour, same order as ^
        self.contours_values_x = []  # the x values for each contour, same order as ^
        self.contours_values_z = [] # the z values for each contour, same order as ^
        self.contours_ax = []  # for each contour idx, what is the axes used        
        self.contours_vmin = []
        self.contours_vmax = []
        
        self.plt = plt  # the pyplot object
        
        assert(0 < nbr_subplots < 3)
        # Currently only considering two simultaneous plots
        if 1 < nbr_subplots < 3:
            if self.subplot_alignment == 'horizontal':
                if list(data_per_subplot[0]['xvalues'].keys())[0] == list(data_per_subplot[1]['xvalues'].keys())[0]:
                    self.fig, axarr = self.plt.subplots(
                            nrows=1, ncols=nbr_subplots,  
                            sharex=True, figsize=figsize)
                    #self.plt.tight_layout()
                    #self.plt.tight_layout(pad=1.2, h_pad=1.2, w_pad=1.2)
                else:
                    self.fig, axarr = self.plt.subplots(
                            nrows=1, ncols=nbr_subplots,  
                            figsize=figsize)
                    #self.plt.tight_layout()
                    #self.plt.tight_layout(pad=1.2, h_pad=1.2, w_pad=1.2)
            else:
                if list(data_per_subplot[0]['xvalues'].keys())[0] == list(data_per_subplot[1]['xvalues'].keys())[0]:
                    self.fig, axarr = self.plt.subplots(
                            nrows=nbr_subplots, ncols=1,  
                            sharex=True, figsize=figsize)
                    self.plt.tight_layout()
                    #self.plt.tight_layout(pad=1.2, h_pad=1.2, w_pad=1.2)
                else:
                    self.fig, axarr = self.plt.subplots(
                            nrows=nbr_subplots, ncols=1,  
                            figsize=figsize)
                    #self.plt.tight_layout()
                    #self.plt.tight_layout(pad=1.2, h_pad=1.2, w_pad=1.2)
            # self.fig.tight_layout(pad=0, h_pad=1.2, w_pad=1.2)
                
        else:
            self.fig = self.plt.figure(figsize=figsize)
            ax = self.plt.axes()
            axarr = [ax]
            #self.plt.tight_layout()

        # self.fig.suptitle("super title")
        # TODO: ^ not supported yet, would need to fix typography

        if titles_per_subplot is not None:
            assert(len(titles_per_subplot) == nbr_subplots)
            for ia, ax in enumerate(axarr):
                ax.set_title(titles_per_subplot[ia])

        for ax in axarr:
            ax.figure.autofmt_xdate()

        self.annotations = defaultdict(lambda: defaultdict(bool))
        # ^ the actual annotation objects, different from the meta
        # ^ nested defaultdict, first key line_name, second index
        # ^ this is in a structured format so we can find each annotation
        self.annotations_li = []
        # ^ same but in list format so we can unpack it later in returns
        # mainly (only?) used for returning in the update methods
        self.annotation_remove = []
        # ^ list of integers for each annotation to keep track when the
        # annotation should be removed. it is a counter that when it becomes
        # zero, the annotation should be removed
        
        # Keeping track of scatter points
        self.annotation_points = []
        self.annotation_points_visible = []

        self.annotation_show_row_nbr_ordered = []
        # the row number in the dataframe of annotations to become visible,
        # ordered
        
        self.hlines = defaultdict(lambda: defaultdict(bool))
        self.hlines_li = []
        self.hlines_remove = []
        self.hlines_show_row_nbr_ordered = []
        self.hlines_annotation = defaultdict(lambda: defaultdict(bool))
        self.hlines_annotation_li = []
        self.hlines_annotation_remove = []
        self.hlines_annotation_show_row_nbr_ordered = []
        self.all_annotations_li = []

        # columns_lhs = list of columns per subplot
        for plot_idx, columns_lhs in enumerate(data_per_subplot):
            # for each list of columns for each subplot
            # i.e. columns_lhs is a list of columns in this subplot
            # plot_idx is the index of the subplot, e.g. 1, 2, 3...
            y = None  # data for first line in subplot placeholder
            z = None  # data for second line in subplot placeholder
            if len(columns_lhs['yvalues']) > 0:
                name = list(columns_lhs['yvalues'].keys())[0]
                x = columns_lhs['xvalues'][list(columns_lhs['xvalues'].keys())[0]]
                y = columns_lhs['yvalues'][list(columns_lhs['yvalues'].keys())[0]]  
                # y = first line in this plot
                # ^ numpy.ndarray
                
                ### Assert that length of x and y matches that of time series
                
                y_series = columns_lhs['yvalues'][list(columns_lhs['yvalues'].keys())[0]]
                y_type = columns_lhs['type_graph'][list(columns_lhs['type_graph'].keys())[0]]
                if y_type == 'contour':
                    z_values = columns_lhs['zvalues'][list(columns_lhs['zvalues'].keys())[0]]
                    assert(len(self.time_series) == z_values.shape[2])
                else:
                    assert(len(self.time_series) == len(x))
                    assert(len(self.time_series) == len(y))
                
                ax = axarr[plot_idx]
                # fmtyaxis(self.plt, ax, fmt_milnodec)
                if y_type == 'expanding_line':
                    self.lines_x_values.append(x)
                    self.lines_y_values.append(y)
                    self.lines_ax.append(ax)
                    self.lines_type.append(y_type)
                    self.line_names.append(name)
                elif y_type == 'changing_line':
                    self.lines_x_values.append(x)
                    self.lines_y_values.append(y)
                    self.lines_ax.append(ax)
                    self.lines_type.append(y_type)
                    self.line_names.append(name)
                elif y_type == 'scatter':
                    self.scatters_values_x.append(x)
                    self.scatters_values_y.append(y)
                    self.scatters_ax.append(ax)
                    self.scatter_names.append(name)
                elif y_type == 'contour':
                    self.contour_names.append(name)
                    self.contours_ax.append(ax)
                    self.contours_values_x.append(x)
                    self.contours_values_y.append(y)
                    self.contours_values_z.append(z_values)
                    self.contours_vmax.append(np.max(z_values))
                    self.contours_vmin.append(np.min(z_values))

                ### for axis limits
                if y_type == 'expanding_line':
                    y_lim_lhs_max = np.max(y_series)
                    y_lim_lhs_min = np.min(y_series)
                    x_lim_lhs_max = np.max(x)
                    x_lim_lhs_min = np.min(x)
                if y_type == 'changing_line' or y_type == 'scatter':
                    max_x_at_t = []
                    min_x_at_t = []
                    max_y_at_t = []
                    min_y_at_t = []
                    for j in range(len(y_series)):
                        max_y_at_t.append(np.max(y_series[j]))
                        min_y_at_t.append(np.min(y_series[j]))
                        max_x_at_t.append(np.max(x[j]))
                        min_x_at_t.append(np.min(x[j]))
                    y_lim_lhs_max = np.max(max_y_at_t)
                    y_lim_lhs_min = np.min(min_y_at_t)
                    x_lim_lhs_max = np.max(max_x_at_t)
                    x_lim_lhs_min = np.min(min_x_at_t)
                if y_type == 'contour':
                    y_lim_lhs_max = np.max(y)
                    y_lim_lhs_min = np.min(y)
                    x_lim_lhs_max = np.max(x)
                    x_lim_lhs_min = np.min(x)
                    z_lim_lhs_min = np.min(z_values)
                    z_lim_lhs_max = np.max(z_values)
                
                if name in self.annotations_meta:
                    annotation_meta_dict = \
                        self.annotations_meta[name]
                else:
                    annotation_meta_dict = {}
                for k in sorted(annotation_meta_dict.keys()):
                    # k = index of dataframe
                    # ki = position as int of index
                    try:
                        ki = np.where(self.time_series == k)[0][0]
                        #ki = df.index.get_loc(k)
                    except Exception as e:
                        raise Exception('Error: ' + str(k) +
                                        ' not in index of time series array')

                    annotation_details_dict = annotation_meta_dict[k]
                    # print(name + ', creating annotation:')
                    # print(annotation_details_dict)
                    if 'x_offset' not in annotation_details_dict:
                        annotation_details_dict['x_offset'] = 0
                    if 'y_offset' not in annotation_details_dict:
                        annotation_details_dict['y_offset'] = 0
                    if 'visible_rows' not in annotation_details_dict:
                        annotation_details_dict['visible_rows'] = 0
                    if 'arrow' not in annotation_details_dict:
                        annotation_details_dict['arrow'] = 'yes'
                    if 'scatter_point' not in annotation_details_dict:
                        annotation_details_dict['scatter_point'] = 'no'
                    
                    ### Need to take more care with point to x/y values
                    if y_type == 'expanding_line':
                        if 'point_to_date' not in annotation_details_dict:
                            annotation_details_dict['point_to_date'] = ki
                        if annotation_details_dict['point_to_date'] != ki:
                            try:
                                ki = np.where(x == annotation_details_dict['point_to_date'])[0][0]
                                #ki = df.index.get_loc(annotation_details_dict['point_to_date'])
                            except Exception as e:
                                ki = np.where(self.time_series == k)[0][0]
                                pass
                        xcoord = x[ki]
                        ycoord = y[ki]
                    elif y_type == 'scatter' or y_type == 'changing_line' or y_type == 'contour':
                        if 'data_point_index_or_data' not in annotation_details_dict:
                            raise Exception('Error: ' + str(name) + str(k) +
                                            'for scatter plot/changing line require data_point_index_or_data input')
                        if 'point_xvalue' not in annotation_details_dict:
                            raise Exception('Error: ' + str(name) + str(k) +
                                            'for scatter plot/changing line require point_xvalue input')
                        if 'point_yvalue' not in annotation_details_dict:
                            raise Exception('Error: ' + str(name) + str(k) +
                                            'for scatter plot/changing line require point_yvalue input')
                        if annotation_details_dict['data_point_index_or_data'] == 'data_point_index':
                            assert(isinstance(annotation_details_dict['point_xvalue'], int) is True)
                            assert(isinstance(annotation_details_dict['point_yvalue'], int) is True)
                            xcoord = x[ki][annotation_details_dict['point_xvalue']]
                            ycoord = y[ki][annotation_details_dict['point_yvalue']]
                        elif annotation_details_dict['data_point_index_or_data'] == 'data':
                            xcoord = annotation_details_dict['point_xvalue']
                            ycoord = annotation_details_dict['point_yvalue']
                        # Already have a scatter point, why need more? Also is it necessary for changing line?
                        annotation_details_dict['scatter_point'] = 'no'
                    
                    if annotation_details_dict['arrow'] == 'yes':
                        a = ax.annotate(
                                annotation_details_dict['text'],
                                xy=(xcoord, ycoord),
                                xycoords='data',
                                xytext=(annotation_details_dict['x_offset'],
                                        annotation_details_dict['y_offset']),
                                arrowprops=dict(arrowstyle='->'),
                                textcoords='offset points')
                    else:
                        a = ax.annotate(
                                annotation_details_dict['text'],
                                xy=(xcoord, ycoord),
                                xycoords='data',
                                xytext=(annotation_details_dict['x_offset'],
                                        annotation_details_dict['y_offset']),
                                textcoords='offset points')
                    a.set_visible(False)
                    self.annotations[name][k] = a
                    self.annotations_li.append(a)
                    self.annotation_show_row_nbr_ordered.append(
                        np.where(self.time_series == k)[0][0])
                    self.annotation_remove.append(-1)  # TODO -1 or 0
                    annot_idx = len(self.annotations_li) - 1
                    annotation_details_dict['__annot_idx'] = annot_idx
                    a.set_animated(True)
                    
                    # Scatter point
                    if annotation_details_dict['scatter_point'] == 'yes':
                        self.annotation_points.append([xcoord, ycoord])
                        self.annotation_points_visible.append([False, annot_idx, plot_idx])
                    
                # Horizontal lines
                if name in self.hlines_meta:
                    hlines_meta_dict = \
                        self.hlines_meta[name]
                else:
                    hlines_meta_dict = {}
                for k in sorted(hlines_meta_dict.keys()):
                    # k = index of dataframe
                    # ki = position as int of index
                    try:
                        ki = np.where(self.time_series == k)[0][0]
                        #ki = df.index.get_loc(k)
                    except Exception as e:
                        raise Exception('Error: ' + str(k) +
                                        ' not in index of time series array')

                    hlines_details_dict = hlines_meta_dict[k]
                    if 'xmin' not in hlines_details_dict:
                        raise Exception('Error: xmin not present. Missing starting point of horizontal line')
                    if 'xmax' not in hlines_details_dict:
                        raise Exception('Error: xmax not present. Missing ending point of horizontal line')
                    if 'xtext' not in hlines_details_dict:
                        raise Exception('Error: xtext not present. Missing x-coordinate for text annotation of horizontal line')

                    # xmin, xmax and ylim have to be provided. Could try and locate points on scatter plot
                    # but that would be inefficient in terms of coding. Otherwise FIX ME.

#                    try:
#                        xmin = np.where(self.time_series == hlines_details_dict['xmin'])[0][0]
#                    except Exception as e:
#                        raise Exception('Error: ' + str(hlines_details_dict['xmin']) +
#                                        ' not in time series of data given.')
#                    try:
#                        xmax = np.where(self.time_series == hlines_details_dict['xmax'])[0][0]
#                    except Exception as e:
#                        raise Exception('Error: ' + str(hlines_details_dict['xmax']) +
#                                        ' not in time series of data given.')
                    xmin = hlines_details_dict['xmin']
                    xmax = hlines_details_dict['xmax']
                    xtext = hlines_details_dict['xtext']
                    if 'yval' not in hlines_details_dict:
                        raise Exception('Error: yval not present. Missing y coordinate of horizontal line')
                    if 'x_offset' not in hlines_details_dict:
                        hlines_details_dict['x_offset'] = 0
                    if 'y_offset' not in hlines_details_dict:
                        hlines_details_dict['y_offset'] = 0
                    if 'visible_rows' not in hlines_details_dict:
                        hlines_details_dict['visible_rows'] = 0
                    hline = ax.annotate(
                            "",
                            xy=(xmin, hlines_details_dict['yval']),
                            xycoords='data',
                            xytext=(xmax, hlines_details_dict['yval']),
                            textcoords='data',
                            va="center", ha="center",
                            arrowprops=dict(arrowstyle="|-|",
                                            connectionstyle="arc3,rad=0"))
                    a = ax.annotate(
                        hlines_details_dict['text'],
                        xy=(xtext, hlines_details_dict['yval']),
                        xycoords='data',
                        xytext=(hlines_details_dict['x_offset'],
                                hlines_details_dict['y_offset']),
                        arrowprops=dict(arrowstyle='->'),
                        textcoords='offset points')
                    hline.set_visible(False)
                    a.set_visible(False)
                    self.hlines[name][k] = hline
                    self.hlines_annotation[name][k] = a
                    self.hlines_li.append(hline)
                    self.hlines_annotation_li.append(a)
                    self.hlines_show_row_nbr_ordered.append(
                        np.where(self.time_series == k)[0][0])
                    self.hlines_annotation_show_row_nbr_ordered.append(
                        np.where(self.time_series == k)[0][0])
                    self.hlines_remove.append(-1)  # TODO -1 or 0
                    self.hlines_annotation_remove.append(-1)
                    annot_idx = len(self.hlines_li) - 1
                    hlines_details_dict['__annot_idx'] = annot_idx
                    a.set_animated(True)
                    hline.set_animated(True)

            if len(columns_lhs['yvalues']) > 1:
                name = list(columns_lhs['yvalues'].keys())[1]
                x = columns_lhs['xvalues'][list(columns_lhs['xvalues'].keys())[1]]
                z = columns_lhs['yvalues'][list(columns_lhs['yvalues'].keys())[1]]  # numpy.ndarray
                z_series = columns_lhs['yvalues'][list(columns_lhs['yvalues'].keys())[1]]  # pandas Series
                z_type = columns_lhs['type_graph'][list(columns_lhs['type_graph'].keys())[1]]
                
                ax = axarr[plot_idx]
                if z_type == 'contour':
                    z_values = columns_lhs['zvalues'][list(columns_lhs['zvalues'].keys())[1]]
                    assert(len(self.time_series) == z_values.shape[2])
                else:
                    assert(len(self.time_series) == len(x))
                    assert(len(self.time_series) == len(z))
                if z_type == 'expanding_line':
                    self.lines_x_values.append(x)
                    self.lines_y_values.append(z)
                    self.lines_ax.append(ax)
                    self.lines_type.append(z_type)
                    self.line_names.append(name)
                elif z_type == 'changing_line':
                    self.lines_x_values.append(x)
                    self.lines_y_values.append(z)
                    self.lines_ax.append(ax)
                    self.lines_type.append(z_type)
                    self.line_names.append(name)
                elif z_type == 'scatter':
                    self.scatters_values_x.append(x)
                    self.scatters_values_y.append(z)
                    self.scatters_ax.append(ax)
                    self.scatter_names.append(name)
                elif z_type == 'contour':
                    self.contour_names.append(name)
                    self.contours_ax.append(ax)
                    self.contours_values_x.append(x)
                    self.contours_values_y.append(z)
                    self.contours_values_z.append(z_values)
                
                ### for axis limits
                if z_type == 'expanding_line':
                    z_lim_lhs_max = np.max(y_series)
                    z_lim_lhs_min = np.min(y_series)
                    x1_lim_lhs_max = np.max(x)
                    x1_lim_lhs_min = np.min(x)
                if z_type == 'changing_line' or z_type == 'scatter':
                    max_x_at_t = []
                    min_x_at_t = []
                    max_z_at_t = []
                    min_z_at_t = []
                    for j in range(len(y_series)):
                        max_z_at_t.append(np.max(z_series[j]))
                        min_z_at_t.append(np.min(z_series[j]))
                        max_x_at_t.append(np.max(x[j]))
                        min_x_at_t.append(np.min(x[j]))
                    z_lim_lhs_max = np.max(max_z_at_t)
                    z_lim_lhs_min = np.min(min_z_at_t)
                    x1_lim_lhs_max = np.max(max_x_at_t)
                    x1_lim_lhs_min = np.min(min_x_at_t)
                if z_type == 'contour':
                    y_lim_lhs_max = np.max(z)
                    y_lim_lhs_min = np.min(z)
                    x_lim_lhs_max = np.max(x)
                    x_lim_lhs_min = np.min(x)
                    z_lim_lhs_min = np.min(z_values)
                    z_lim_lhs_max = np.max(z_values)
                
                ax = axarr[plot_idx]
                if name in self.annotations_meta:
                    annotation_meta_dict = \
                        self.annotations_meta[name]
                else:
                    annotation_meta_dict = {}
                for k in sorted(annotation_meta_dict.keys()):
                    # k = index of dataframe
                    # ki = position as int of index
                    try:
                        ki = np.where(self.time_series == k)[0][0]
                        #ki = df.index.get_loc(k)
                    except Exception as e:
                        raise Exception('Error: ' + str(k) +
                                        ' not in index of time series array')

                    annotation_details_dict = annotation_meta_dict[k]
                    # print(name + ', creating annotation:')
                    # print(annotation_details_dict)
                    if 'x_offset' not in annotation_details_dict:
                        annotation_details_dict['x_offset'] = 0
                    if 'y_offset' not in annotation_details_dict:
                        annotation_details_dict['y_offset'] = 0
                    if 'visible_rows' not in annotation_details_dict:
                        annotation_details_dict['visible_rows'] = 0
                    if 'arrow' not in annotation_details_dict:
                        annotation_details_dict['arrow'] = 'yes'
                    if 'scatter_point' not in annotation_details_dict:
                        annotation_details_dict['scatter_point'] = 'no'
                    
                    ### Need to take more care with point to x/y values
                    if z_type == 'expanding_line':
                        if 'point_to_date' not in annotation_details_dict:
                            annotation_details_dict['point_to_date'] = ki
                        if annotation_details_dict['point_to_date'] != ki:
                            try:
                                ki = np.where(x == annotation_details_dict['point_to_date'])[0][0]
                                #ki = df.index.get_loc(annotation_details_dict['point_to_date'])
                            except Exception as e:
                                ki = np.where(self.time_series == k)[0][0]
                                pass
                        xcoord = x[ki]
                        ycoord = z[ki]
                    elif z_type == 'scatter' or z_type == 'changing_line' or z_type == 'contour':
                        if 'data_point_index_or_data' not in annotation_details_dict:
                            raise Exception('Error: ' + str(name) + str(k) +
                                            'for scatter plot/changing line require data_point_index_or_data input')
                        if 'point_xvalue' not in annotation_details_dict:
                            raise Exception('Error: ' + str(name) + str(k) +
                                            'for scatter plot/changing line require point_xvalue input')
                        if 'point_yvalue' not in annotation_details_dict:
                            raise Exception('Error: ' + str(name) + str(k) +
                                            'for scatter plot/changing line require point_yvalue input')
                        if annotation_details_dict['data_point_index_or_data'] == 'data_point_index':
                            assert(isinstance(annotation_details_dict['point_xvalue'], int) is True)
                            assert(isinstance(annotation_details_dict['point_yvalue'], int) is True)
                            xcoord = x[ki][annotation_details_dict['point_xvalue']]
                            ycoord = z[ki][annotation_details_dict['point_yvalue']]
                        elif annotation_details_dict['data_point_index_or_data'] == 'data':
                            xcoord = annotation_details_dict['point_xvalue']
                            ycoord = annotation_details_dict['point_yvalue']
                        # Already have a scatter point, why need more? Similarly for changing line
                        annotation_details_dict['scatter_point'] = 'no'
                    
                    if annotation_details_dict['arrow'] == 'yes':
                        a = ax.annotate(
                                annotation_details_dict['text'],
                                xy=(xcoord, ycoord),
                                xycoords='data',
                                xytext=(annotation_details_dict['x_offset'],
                                        annotation_details_dict['y_offset']),
                                arrowprops=dict(arrowstyle='->'),
                                textcoords='offset points')
                    else:
                        a = ax.annotate(
                                annotation_details_dict['text'],
                                xy=(xcoord, ycoord),
                                xycoords='data',
                                xytext=(annotation_details_dict['x_offset'],
                                        annotation_details_dict['y_offset']),
                                textcoords='offset points')
                    a.set_visible(False)
                    self.annotations[name][k] = a
                    self.annotations_li.append(a)
                    self.annotation_show_row_nbr_ordered.append(
                        np.where(self.time_series == k)[0][0])
                    self.annotation_remove.append(-1)  # TODO -1 or 0
                    annot_idx = len(self.annotations_li) - 1
                    annotation_details_dict['__annot_idx'] = annot_idx
                    a.set_animated(True)
                    
                    # Scatter point
                    if annotation_details_dict['scatter_point'] == 'yes':
                        self.annotation_points.append([xcoord, ycoord])
                        self.annotation_points_visible.append([False, annot_idx, plot_idx])
                    
                # Horizontal lines
                if name in self.hlines_meta:
                    hlines_meta_dict = \
                        self.hlines_meta[name]
                else:
                    hlines_meta_dict = {}
                for k in sorted(hlines_meta_dict.keys()):
                    # k = index of dataframe
                    # ki = position as int of index
                    try:
                        ki = np.where(self.time_series == k)[0][0]
                        #ki = df.index.get_loc(k)
                    except Exception as e:
                        raise Exception('Error: ' + str(k) +
                                        ' not in index of time series array')

                    hlines_details_dict = hlines_meta_dict[k]
                    if 'xmin' not in hlines_details_dict:
                        raise Exception('Error: xmin not present. Missing starting point of horizontal line')
                    if 'xmax' not in hlines_details_dict:
                        raise Exception('Error: xmax not present. Missing ending point of horizontal line')
                    if 'xtext' not in hlines_details_dict:
                        raise Exception('Error: xtext not present. Missing x-coordinate for text annotation of horizontal line')

                    # xmin, xmax and ylim have to be provided. Could try and locate points on scatter plot
                    # but that would be inefficient in terms of coding. Otherwise FIX ME.

#                    try:
#                        xmin = np.where(self.time_series == hlines_details_dict['xmin'])[0][0]
#                    except Exception as e:
#                        raise Exception('Error: ' + str(hlines_details_dict['xmin']) +
#                                        ' not in time series of data given.')
#                    try:
#                        xmax = np.where(self.time_series == hlines_details_dict['xmax'])[0][0]
#                    except Exception as e:
#                        raise Exception('Error: ' + str(hlines_details_dict['xmax']) +
#                                        ' not in time series of data given.')
                    xmin = hlines_details_dict['xmin']
                    xmax = hlines_details_dict['xmax']
                    xtext = hlines_details_dict['xtext']
                    if 'yval' not in hlines_details_dict:
                        raise Exception('Error: yval not present. Missing y coordinate of horizontal line')
                    if 'x_offset' not in hlines_details_dict:
                        hlines_details_dict['x_offset'] = 0
                    if 'y_offset' not in hlines_details_dict:
                        hlines_details_dict['y_offset'] = 0
                    if 'visible_rows' not in hlines_details_dict:
                        hlines_details_dict['visible_rows'] = 0
                    hline = ax.annotate(
                            "",
                            xy=(xmin, hlines_details_dict['yval']),
                            xycoords='data',
                            xytext=(xmax, hlines_details_dict['yval']),
                            textcoords='data',
                            va="center", ha="center",
                            arrowprops=dict(arrowstyle="|-|",
                                            connectionstyle="arc3,rad=0"))
                    a = ax.annotate(
                        hlines_details_dict['text'],
                        xy=(xtext, hlines_details_dict['yval']),
                        xycoords='data',
                        xytext=(hlines_details_dict['x_offset'],
                                hlines_details_dict['y_offset']),
                        arrowprops=dict(arrowstyle='->'),
                        textcoords='offset points')
                    hline.set_visible(False)
                    a.set_visible(False)
                    self.hlines[name][k] = hline
                    self.hlines_annotation[name][k] = a
                    self.hlines_li.append(hline)
                    self.hlines_annotation_li.append(a)
                    self.hlines_show_row_nbr_ordered.append(
                        np.where(self.time_series == k)[0][0])
                    self.hlines_annotation_show_row_nbr_ordered.append(
                        np.where(self.time_series == k)[0][0])
                    self.hlines_remove.append(-1)  # TODO -1 or 0
                    self.hlines_annotation_remove.append(-1)
                    annot_idx = len(self.hlines_li) - 1
                    hlines_details_dict['__annot_idx'] = annot_idx
                    a.set_animated(True)
                    hline.set_animated(True)

            if len(columns_lhs['yvalues']) > 2:
                raise NotImplemented("Max two lines per plot is supported.")

            ax = axarr[plot_idx]
            for idx, c in enumerate(list(columns_lhs['type_graph'].keys())):
                if columns_lhs['type_graph'][c] == 'expanding_line' or columns_lhs['type_graph'][c] == 'changing_line':
                    l, = ax.plot([], [], lw=1)
                    self.lines.append(l)
                elif columns_lhs['type_graph'][c] == 'scatter':
                    scat = ax.scatter([], [], alpha=1, s = 5)
                    self.scatters.append(scat)
                elif columns_lhs['type_graph'][c] == 'contour':
                    contour = ax.contourf([0,0], [0,0], [[0,0], [0,0]], 100, 
                                          vmin = np.min(columns_lhs['zvalues'][list(columns_lhs['zvalues'].keys())[idx]]), 
                                          vmax = np.max(columns_lhs['zvalues'][list(columns_lhs['zvalues'].keys())[idx]])) 
                    self.contours.append(contour)
            if z is not None:
                y_lim_lhs_max = max(y_lim_lhs_max, z_lim_lhs_max)
                y_lim_lhs_min = min(y_lim_lhs_min, z_lim_lhs_min)
                x_lim_lhs_max = max(x_lim_lhs_max, x1_lim_lhs_max)
                x_lim_lhs_min = min(x_lim_lhs_min, x1_lim_lhs_min)
            
            ylim_min_multiplier = self.YLIM_MIN_MULTIPLIER
            if plot_idx in self.attribs and \
                    'y_min_multiplier' in self.attribs[plot_idx]:
                        ylim_min_multiplier = \
                        self.attribs[plot_idx]['y_min_multiplier']

            ylim_max_multiplier = self.YLIM_MAX_MULTIPLIER
            if plot_idx in self.attribs and \
                    'y_max_multiplier' in self.attribs[plot_idx]:
                        ylim_max_multiplier = \
                        self.attribs[plot_idx]['y_max_multiplier']

            y_lim_lhs_max = ylim_max_multiplier * y_lim_lhs_max
            y_lim_lhs_min = ylim_min_multiplier * y_lim_lhs_min
            # only handle lhs charts for now
            # ylim max is highest of the two lines times a factor
            # ylim min is lowest of the two lines times a factor (can be 0)
            if self.axis_limits is None:
                ax.set_xlim(x_lim_lhs_min, x_lim_lhs_max)
                ax.set_ylim(y_lim_lhs_min, y_lim_lhs_max)
            else:
                ax.set_xlim(self.axis_limits[plot_idx][0][0], self.axis_limits[plot_idx][0][1])
                ax.set_ylim(self.axis_limits[plot_idx][1][0], self.axis_limits[plot_idx][1][1])
                ax.set_xlabel('q')
                if plot_idx == 0:
                    ax.set_ylabel('p')

            if use_legends:
                if legends_rename is None:
                    legends_rename = {}
                legend_names = [legends_rename[c]
                                if c in legends_rename
                                else c
                                for c in list(columns_lhs['yvalues'].keys())]
                ax.legend(legend_names, loc=self.legends_loc)
            
        nbr_lines = len(self.lines)
        self._sleep_counter = 0
        self._annotation_done = defaultdict(lambda: defaultdict(bool))
        # # ^ nested defaultdict, first key line_name, second date
        # ^ removed, is it necessary        
        self._hlines_done = defaultdict(lambda: defaultdict(bool))
        self._hlines_annotation_done = defaultdict(lambda: defaultdict(bool))        
        self.axarr = axarr
        self.annotation_show_row_nbr_ordered.sort()
        self.hlines_show_row_nbr_ordered.sort()
        self.hlines_annotation_show_row_nbr_ordered.sort()
        assert(len(self.lines) == len(self.lines_x_values))
        assert(len(self.scatters) == len(self.scatters_values_x))
        assert(len(self.annotation_remove) == len(self.annotations_li))
        assert(len(self.hlines_remove) == len(self.hlines_li))
        assert(len(self.hlines_annotation_remove) == len(self.hlines_annotation_li))

    def get_number_of_frames_required(self):
        anim = self.get_animate_function()

        # calculate total number of frames
        # because of the speed parameter, instead of trying to
        # calculate it closed form we just loop through the animate function
        # until an -1 is returned
        u = 0
        while True:
            u += 1
            x = anim(u)
            if x == -1:
                if self.DEBUG:
                    print('returning number of frames: ' + str(u - 1))
                self.__init__(
                    time_series=self.time_series,
                    data_per_subplot=self.data_per_subplot,
                    plt=self.plt,
                    axis_limits = self.axis_limits,
                    subplot_alignment=self.subplot_alignment,
                    annotations=self.annotations_given,
                    horizontal_lines=self.h_lines_given,
                    figsize=self.figsize,
                    titles_per_subplot=self.titles_per_subplot,
                    use_legends=self.use_legends,
                    legends_loc=self.legends_loc,
                    legends_rename=self.legends_rename,
                    ylim_min_multiplier_default=\
                        self.ylim_min_multiplier_default,
                    ylim_max_multiplier_default=\
                        self.ylim_max_multiplier_default,
                    speed=self.SPEED,
                    attribs=self.attribs
                    )
                # ^ restore initialisation of values
                return u - 1

        # # "Closed form solution" but does not handle SPEED
        # kept for reference in case SPEED treatment can be handled
        #
        # old version, which works in the abscence of SPEED
        # nbr_rows = len(self.df)
        # nbr_sleep_frames = 0
        # for a in self.annotations_meta:
        #     for k in self.annotations_meta[a]:
        #         # k are the dates
        #         s = self.annotations_meta[a][k]['sleep_frames']
        #         nbr_sleep_frames += s
        # total_frames = nbr_rows + nbr_sleep_frames
        # # print('total nbr_rows: ' + str(nbr_rows))
        # # print('total nbr_sleep_frames: ' + str(nbr_sleep_frames))
        # print('total frames: ' + str(total_frames))
        # return total_frames

    def get_init_function(self):
        def init_animate():
            for l in self.lines:
                # initialise all lines
                l.set_data([], [])
            for scat_idx, scat in enumerate(self.scatters):
                scat.set_offsets([(0,0)])
                
            for contour_idx, contour in enumerate(self.contours):
                ax = self.contours_ax[contour_idx]
                x = self.contours_values_x[contour_idx]
                y = self.contours_values_y[contour_idx]
                z = self.contours_values_z[contour_idx]
                self.contours[contour_idx] = ax.contourf(x, y, z[:, :, 0], 50, cmap='RdBu')
                colorbar = self.fig.colorbar(self.contours[contour_idx], 
                                             ticks = [self.contours_vmin[contour_idx], self.contours_vmax[contour_idx]])
                colorbar.ax.set_yticklabels(['{:.3f}'.format(self.contours_vmin[contour_idx]),'{:.3f}'.format(self.contours_vmax[contour_idx])]) 
                colorbar.set_label(r'$|\psi(q,p)|^2$', rotation = 270, labelpad = -5)
                self.all_annotations_li = self.annotations_li + self.hlines_annotation_li + self.hlines_li 
            return (*self.lines, *self.scatters, *self.contours, *self.all_annotations_li)
        return init_animate

    def get_animate_function(self):
        def animate(i):
            # i is the frame number
            #
            rownbr = i - self.sleep_lag + self.step_size_cumulative
            # ^ this is the relevant row number in the dataframe

            # ---
            # when is the next annotation action?
            # if we have a speed up, we need to make sure we do
            # not miss that action, whether it's an addition or removal
            if self.DEBUG:
                print('---')
                print('animate() called for frame i: ' + str(i))
            if self.DEBUG:
                print('annotation_show_row_nbr_ordered ' +
                      str(self.annotation_show_row_nbr_ordered))
                print('annotation remove countdown: ' +
                      str(self.annotation_remove))
                print('hlines_show_row_nbr_ordered ' +
                      str(self.hlines_show_row_nbr_ordered))
                print('hlines remove countdown: ' +
                      str(self.hlines_remove))
            next_annotation_row = len(self.time_series)
            next_hlines_row = len(self.time_series)
            if len(self.annotation_show_row_nbr_ordered) > 0:
                next_annotation_row = self.annotation_show_row_nbr_ordered[0]
            if len(self.hlines_show_row_nbr_ordered) > 0:
                next_hlines_row = self.hlines_show_row_nbr_ordered[0]
            next_annotation_remove = -1
            next_hlines_remove = -1
            annots_remove_not_minus_one = [w for w
                                           in self.annotation_remove
                                           if w > -1]
            hlines_remove_not_minus_one = [w for w
                                           in self.hlines_remove
                                           if w > -1]
            if len(annots_remove_not_minus_one) > 0:
                next_annotation_remove = min(annots_remove_not_minus_one) + \
                    rownbr
            if len(hlines_remove_not_minus_one) > 0:
                next_hlines_remove = min(hlines_remove_not_minus_one) + \
                    rownbr
            if next_annotation_remove > -1:
                if self.DEBUG:
                    print('next_annotation_remove ' +
                          str(next_annotation_remove))
                next_annotation_row = min(next_annotation_row,
                                          next_annotation_remove)
            print('next_annotation_row: ' + str(next_annotation_row))
            # ^ next_annotation_row is now the next row which contains the
            #   annotation action
            # ---
            if next_hlines_remove > -1:
                if self.DEBUG:
                    print('next_hlines_remove ' +
                          str(next_hlines_remove))
                next_hlines_row = min(next_hlines_row,
                                      next_hlines_remove)
            print('next_hlines_row: ' + str(next_hlines_row))
            
            if self.DEBUG:
                print('rownbr #: ' + str(rownbr))
                print('_sleep_counter: ' + str(self._sleep_counter))
                print('sleep lag: ' + str(self.sleep_lag))
                print('step_size_cumulative: ' +
                      str(self.step_size_cumulative))
            # ^ need to lag it with total sleep time so far
            step_size = 0
            if self._sleep_counter == 0:
                # ^ if not sleeping:
                step_size = min((self.SPEED - 1),
                                max((next_annotation_row - rownbr), 0),
                                max((next_hlines_row - rownbr), 0),
                                len(self.time_series) - rownbr)
                # ^ if we assume animate() is called for each frame in turn
                # SPEED - 1, means a factor of SPEED, e.g. (4-1) + 1 = 4x
                # i.e. step_size is the additional rows to skip each frame
                # assert(rownbr <= len(self.df))
                assert(step_size >= 0)
                if self.DEBUG:
                    print('step_size: ' + str(step_size))
                    # input()
                self.step_size_cumulative += step_size
                rownbr += step_size

                if (rownbr + 1) >= len(self.time_series):
                    if self.DEBUG:
                        print('reached end')
                    return -1

            if self._sleep_counter > 0:
                print('sleeping frame')
                # sleep, nothing to do now, return early
                self._sleep_counter -= 1
                self.all_annotations_li = self.annotations_li + self.hlines_li + self.hlines_annotation_li
                return (*self.lines, *self.scatters, *self.all_annotations_li)
            assert(rownbr + 1 >= 0)
        
            # remove any annotations due to be removed
            for ir, rc in enumerate(self.annotation_remove):
                # TODO does this work if called multiple times per day?
                # rc = remove counter, counts down to 0
                if rc == 0:  # TODO: or 0 ? start with -1?
                    a = self.annotations_li[ir]
                    if self.DEBUG:
                        print('removing annotation: ' + str(a))
                    a.set_visible(False)
                    for j in range(len(self.annotation_points_visible)):
                        if ir == self.annotation_points_visible[j][1]:
                            self.annotation_points_visible[j][0] = False
                self.annotation_remove[ir] = \
                    max(rc - max(step_size, 1), -1)
                    # max(rc - 1, -1)
                # ^ count down annotation remove counter
            
            # remove any horizontal lines due to be removed
            for ir, rc in enumerate(self.hlines_remove):
                # TODO does this work if called multiple times per day?
                # rc = remove counter, counts down to 0
                if rc == 0:  # TODO: or 0 ? start with -1?
                    hline = self.hlines_li[ir]
                    if self.DEBUG:
                        print('removing horizontal line: ' + str(hline))
                    hline.set_visible(False)
                self.hlines_remove[ir] = \
                    max(rc - max(step_size, 1), -1)
                    # max(rc - 1, -1)
                # ^ count down annotation remove counter
            
            # remove any horizontal line annotations due to be removed
            for ir, rc in enumerate(self.hlines_annotation_remove):
                # TODO does this work if called multiple times per day?
                # rc = remove counter, counts down to 0
                if rc == 0:  # TODO: or 0 ? start with -1?
                    hline_annot = self.hlines_annotation_li[ir]
                    if self.DEBUG:
                        print('removing horizontal line annotation: ' + str(hline_annot))
                    hline_annot.set_visible(False)
                self.hlines_annotation_remove[ir] = \
                    max(rc - max(step_size, 1), -1)
                    # max(rc - 1, -1)
                # ^ count down annotation remove counter
            
            for line_idx, l in enumerate(self.lines):
                # line_idx = line_idx of the line
                # plot all lines

                line_name = self.line_names[line_idx]
                line_type = self.lines_type[line_idx]
                x_values = self.lines_x_values[line_idx]
                y_values = self.lines_y_values[line_idx]
                if line_type == 'expanding_line':
                    x = x_values[0:rownbr + 1]
                    y = y_values[0:rownbr + 1]
                elif line_type == 'changing_line':
                    x = x_values[rownbr + 1]
                    y = y_values[rownbr + 1]
                l.set_data(x, y)
                if self.line_names[line_idx] in self.annotations_meta:
                    annotation_dict = \
                        self.annotations_meta[self.line_names[line_idx]]
                else:
                    annotation_dict = {}
                
                if self.line_names[line_idx] in self.hlines_meta:
                    hlines_dict = \
                        self.hlines_meta[self.line_names[line_idx]]
                else:
                    hlines_dict = {}
                
                # print('rownbr here: ' + str(rownbr))
                if rownbr > 0 and self.time_series[rownbr] in annotation_dict and \
                        not self._annotation_done[line_name][self.time_series[rownbr]]:
                    # nb: annotation_done check is necessary, do not remove
                    # as it may be called multiple times per index day
                    # it appears?
                    annotation_details_dict = \
                        annotation_dict[self.time_series[rownbr]]
                    if self.DEBUG:
                        print(line_name + ' displaying annotation:')
                        print(annotation_details_dict)
                    annot = self.annotations[line_name][self.time_series[rownbr]]
                    self._annotation_done[line_name][self.time_series[rownbr]] = True
                    annot.set_visible(True)
                    _rownbr = self.annotation_show_row_nbr_ordered.pop(0)
                    if _rownbr != rownbr:
                        print('_rownbr ' + str(_rownbr))
                        print('rownbr ' + str(rownbr))
                        # should never happen...
                    assert(_rownbr == rownbr)

                    if 'sleep_frames' in annotation_dict[self.time_series[rownbr]]:
                        self._sleep_counter = \
                            annotation_dict[self.time_series[rownbr]]['sleep_frames']
                        self.sleep_lag += self._sleep_counter
                        # ^ TODO: what if multiple sleeps at the same time?
                        # print('sleep lag: ' + str(self.sleep_lag))
                    else:
                        self._sleep_counter = 0

                    annot_idx = annotation_dict[self.time_series[rownbr]]['__annot_idx']
                    self.annotation_remove[annot_idx] = \
                        annotation_details_dict['visible_rows']
                    # ^ count down frames until it should be removed visually
                    
                    # Set scatter point to be visible
                    for j in range(len(self.annotation_points_visible)):
                        if annot_idx == self.annotation_points_visible[j][1]:
                            self.annotation_points_visible[j][0] = True
                
                if rownbr > 0 and self.time_series[rownbr] in hlines_dict and \
                        not self._hlines_done[line_name][self.time_series[rownbr]]:
                    # nb: annotation_done check is necessary, do not remove
                    # as it may be called multiple times per index day
                    # it appears?
                    hlines_details_dict = \
                        hlines_dict[self.time_series[rownbr]]
                    if self.DEBUG:
                        print(line_name + ' displaying horizontal line:')
                        print(hlines_details_dict)
                    hline = self.hlines[line_name][self.time_series[rownbr]]
                    self._hlines_done[line_name][self.time_series[rownbr]] = True
                    hline.set_visible(True)
                    _rownbr = self.hlines_show_row_nbr_ordered.pop(0)
                    if _rownbr != rownbr:
                        print('_rownbr ' + str(_rownbr))
                        print('rownbr ' + str(rownbr))
                        # should never happen...
                    assert(_rownbr == rownbr)

                    if 'sleep_frames' in hlines_dict[self.time_series[rownbr]]:
                        self._sleep_counter = \
                            hlines_dict[self.time_series[rownbr]]['sleep_frames']
                        self.sleep_lag += self._sleep_counter
                        # ^ TODO: what if multiple sleeps at the same time?
                        # print('sleep lag: ' + str(self.sleep_lag))
                    else:
                        self._sleep_counter = 0

                    annot_idx = hlines_dict[self.time_series[rownbr]]['__annot_idx']
                    self.hlines_remove[annot_idx] = \
                        hlines_details_dict['visible_rows']
                    # ^ count down frames until it should be removed visually
                
                if rownbr > 0 and self.time_series[rownbr] in hlines_dict and \
                        not self._hlines_annotation_done[line_name][self.time_series[rownbr]]:
                    # nb: annotation_done check is necessary, do not remove
                    # as it may be called multiple times per index day
                    # it appears?
                    hlines_details_dict = \
                        hlines_dict[self.time_series[rownbr]]
                    if self.DEBUG:
                        print(line_name + ' displaying horizontal line annotation:')
                        print(hlines_details_dict)
                    hline_annot = self.hlines_annotation[line_name][self.time_series[rownbr]]
                    self._hlines_annotation_done[line_name][self.time_series[rownbr]] = True
                    hline_annot.set_visible(True)
                    _rownbr = self.hlines_annotation_show_row_nbr_ordered.pop(0)
                    if _rownbr != rownbr:
                        print('_rownbr ' + str(_rownbr))
                        print('rownbr ' + str(rownbr))
                        # should never happen...
                    assert(_rownbr == rownbr)

                    annot_idx = hlines_dict[self.time_series[rownbr]]['__annot_idx']
                    self.hlines_annotation_remove[annot_idx] = \
                        hlines_details_dict['visible_rows']
                    # ^ count down frames until it should be removed visually
            
            for scatter_idx, scat in enumerate(self.scatters):
                # scatter_idx = scatter_idx of the scatter points
                # plot all scatter points

                scatter_name = self.scatter_names[scatter_idx]
                y_vals = self.scatters_values_y[scatter_idx]
                x_vals = self.scatters_values_x[scatter_idx]
                print(rownbr + 1)
                y = y_vals[rownbr + 1]
                print(y)
                x = x_vals[rownbr + 1]
                assert(len(x) == len(y))
                scat.set_offsets([(x[j], y[j]) for j in range(len(x))])
                
                if self.scatter_names[scatter_idx] in self.annotations_meta:
                    annotation_dict = \
                        self.annotations_meta[self.scatter_names[scatter_idx]]
                else:
                    annotation_dict = {}
                
                if self.scatter_names[scatter_idx] in self.hlines_meta:
                    hlines_dict = \
                        self.hlines_meta[self.scatter_names[scatter_idx]]
                else:
                    hlines_dict = {}
                
                # print('rownbr here: ' + str(rownbr))
                if rownbr > 0 and self.time_series[rownbr] in annotation_dict and \
                        not self._annotation_done[scatter_name][self.time_series[rownbr]]:
                    # nb: annotation_done check is necessary, do not remove
                    # as it may be called multiple times per index day
                    # it appears?
                    annotation_details_dict = \
                        annotation_dict[self.time_series[rownbr]]
                    if self.DEBUG:
                        print(scatter_name + ' displaying annotation:')
                        print(annotation_details_dict)
                    annot = self.annotations[scatter_name][self.time_series[rownbr]]
                    self._annotation_done[scatter_name][self.time_series[rownbr]] = True
                    annot.set_visible(True)
                    _rownbr = self.annotation_show_row_nbr_ordered.pop(0)
                    if _rownbr != rownbr:
                        print('_rownbr ' + str(_rownbr))
                        print('rownbr ' + str(rownbr))
                        # should never happen...
                    assert(_rownbr == rownbr)

                    if 'sleep_frames' in annotation_dict[self.time_series[rownbr]]:
                        self._sleep_counter = \
                            annotation_dict[self.time_series[rownbr]]['sleep_frames']
                        self.sleep_lag += self._sleep_counter
                        # ^ TODO: what if multiple sleeps at the same time?
                        # print('sleep lag: ' + str(self.sleep_lag))
                    else:
                        self._sleep_counter = 0

                    annot_idx = annotation_dict[self.time_series[rownbr]]['__annot_idx']
                    self.annotation_remove[annot_idx] = \
                        annotation_details_dict['visible_rows']
                    # ^ count down frames until it should be removed visually
                    
                if rownbr > 0 and self.time_series[rownbr] in hlines_dict and \
                        not self._hlines_done[scatter_name][self.time_series[rownbr]]:
                    # nb: annotation_done check is necessary, do not remove
                    # as it may be called multiple times per index day
                    # it appears?
                    hlines_details_dict = \
                        hlines_dict[self.time_series[rownbr]]
                    if self.DEBUG:
                        print(scatter_name + ' displaying horizontal line:')
                        print(hlines_details_dict)
                    hline = self.hlines[scatter_name][self.time_series[rownbr]]
                    self._hlines_done[scatter_name][self.time_series[rownbr]] = True
                    hline.set_visible(True)
                    _rownbr = self.hlines_show_row_nbr_ordered.pop(0)
                    if _rownbr != rownbr:
                        print('_rownbr ' + str(_rownbr))
                        print('rownbr ' + str(rownbr))
                        # should never happen...
                    assert(_rownbr == rownbr)

                    if 'sleep_frames' in hlines_dict[self.time_series[rownbr]]:
                        self._sleep_counter = \
                            hlines_dict[self.time_series[rownbr]]['sleep_frames']
                        self.sleep_lag += self._sleep_counter
                        # ^ TODO: what if multiple sleeps at the same time?
                        # print('sleep lag: ' + str(self.sleep_lag))
                    else:
                        self._sleep_counter = 0

                    annot_idx = hlines_dict[self.time_series[rownbr]]['__annot_idx']
                    self.hlines_remove[annot_idx] = \
                        hlines_details_dict['visible_rows']
                    # ^ count down frames until it should be removed visually
                
                if rownbr > 0 and self.time_series[rownbr] in hlines_dict and \
                        not self._hlines_annotation_done[scatter_name][self.time_series[rownbr]]:
                    # nb: annotation_done check is necessary, do not remove
                    # as it may be called multiple times per index day
                    # it appears?
                    hlines_details_dict = \
                        hlines_dict[self.time_series[rownbr]]
                    if self.DEBUG:
                        print(scatter_name + ' displaying horizontal line annotation:')
                        print(hlines_details_dict)
                    hline_annot = self.hlines_annotation[scatter_name][self.time_series[rownbr]]
                    self._hlines_annotation_done[scatter_name][self.time_series[rownbr]] = True
                    hline_annot.set_visible(True)
                    _rownbr = self.hlines_annotation_show_row_nbr_ordered.pop(0)
                    if _rownbr != rownbr:
                        print('_rownbr ' + str(_rownbr))
                        print('rownbr ' + str(rownbr))
                        # should never happen...
                    assert(_rownbr == rownbr)

                    annot_idx = hlines_dict[self.time_series[rownbr]]['__annot_idx']
                    self.hlines_annotation_remove[annot_idx] = \
                        hlines_details_dict['visible_rows']
                    # ^ count down frames until it should be removed visually
            
            for contour_idx, contour in enumerate(self.contours):
                # scatter_idx = scatter_idx of the scatter points
                # plot all scatter points

                contour_name = self.contour_names[contour_idx]
                y = self.contours_values_y[contour_idx]
                x = self.contours_values_x[contour_idx]
                z_vals = self.contours_values_z[contour_idx]
                z = z_vals[:, :, rownbr + 1]
                ax = self.contours_ax[contour_idx]
                self.contours[contour_idx] = ax.contourf(x, y, z, 50, cmap='RdBu', 
                             vmin = self.contours_vmin[contour_idx],
                             vmax = self.contours_vmax[contour_idx])
                
                if self.contour_names[contour_idx] in self.annotations_meta:
                    annotation_dict = \
                        self.annotations_meta[self.contour_names[contour_idx]]
                else:
                    annotation_dict = {}
                
                if self.contour_names[contour_idx] in self.hlines_meta:
                    hlines_dict = \
                        self.hlines_meta[self.contour_names[contour_idx]]
                else:
                    hlines_dict = {}
                
                # print('rownbr here: ' + str(rownbr))
                if rownbr > 0 and self.time_series[rownbr] in annotation_dict and \
                        not self._annotation_done[contour_name][self.time_series[rownbr]]:
                    # nb: annotation_done check is necessary, do not remove
                    # as it may be called multiple times per index day
                    # it appears?
                    annotation_details_dict = \
                        annotation_dict[self.time_series[rownbr]]
                    if self.DEBUG:
                        print(scatter_name + ' displaying annotation:')
                        print(annotation_details_dict)
                    annot = self.annotations[contour_name][self.time_series[rownbr]]
                    self._annotation_done[contour_name][self.time_series[rownbr]] = True
                    annot.set_visible(True)
                    _rownbr = self.annotation_show_row_nbr_ordered.pop(0)
                    if _rownbr != rownbr:
                        print('_rownbr ' + str(_rownbr))
                        print('rownbr ' + str(rownbr))
                        # should never happen...
                    assert(_rownbr == rownbr)

                    if 'sleep_frames' in annotation_dict[self.time_series[rownbr]]:
                        self._sleep_counter = \
                            annotation_dict[self.time_series[rownbr]]['sleep_frames']
                        self.sleep_lag += self._sleep_counter
                        # ^ TODO: what if multiple sleeps at the same time?
                        # print('sleep lag: ' + str(self.sleep_lag))
                    else:
                        self._sleep_counter = 0

                    annot_idx = annotation_dict[self.time_series[rownbr]]['__annot_idx']
                    self.annotation_remove[annot_idx] = \
                        annotation_details_dict['visible_rows']
                    # ^ count down frames until it should be removed visually
                    
                if rownbr > 0 and self.time_series[rownbr] in hlines_dict and \
                        not self._hlines_done[contour_name][self.time_series[rownbr]]:
                    # nb: annotation_done check is necessary, do not remove
                    # as it may be called multiple times per index day
                    # it appears?
                    hlines_details_dict = \
                        hlines_dict[self.time_series[rownbr]]
                    if self.DEBUG:
                        print(scatter_name + ' displaying horizontal line:')
                        print(hlines_details_dict)
                    hline = self.hlines[contour_name][self.time_series[rownbr]]
                    self._hlines_done[contour_name][self.time_series[rownbr]] = True
                    hline.set_visible(True)
                    _rownbr = self.hlines_show_row_nbr_ordered.pop(0)
                    if _rownbr != rownbr:
                        print('_rownbr ' + str(_rownbr))
                        print('rownbr ' + str(rownbr))
                        # should never happen...
                    assert(_rownbr == rownbr)

                    if 'sleep_frames' in hlines_dict[self.time_series[rownbr]]:
                        self._sleep_counter = \
                            hlines_dict[self.time_series[rownbr]]['sleep_frames']
                        self.sleep_lag += self._sleep_counter
                        # ^ TODO: what if multiple sleeps at the same time?
                        # print('sleep lag: ' + str(self.sleep_lag))
                    else:
                        self._sleep_counter = 0

                    annot_idx = hlines_dict[self.time_series[rownbr]]['__annot_idx']
                    self.hlines_remove[annot_idx] = \
                        hlines_details_dict['visible_rows']
                    # ^ count down frames until it should be removed visually
                
                if rownbr > 0 and self.time_series[rownbr] in hlines_dict and \
                        not self._hlines_annotation_done[contour_name][self.time_series[rownbr]]:
                    # nb: annotation_done check is necessary, do not remove
                    # as it may be called multiple times per index day
                    # it appears?
                    hlines_details_dict = \
                        hlines_dict[self.time_series[rownbr]]
                    if self.DEBUG:
                        print(contour_name + ' displaying horizontal line annotation:')
                        print(hlines_details_dict)
                    hline_annot = self.hlines_annotation[contour_name][self.time_series[rownbr]]
                    self._hlines_annotation_done[contour_name][self.time_series[rownbr]] = True
                    hline_annot.set_visible(True)
                    _rownbr = self.hlines_annotation_show_row_nbr_ordered.pop(0)
                    if _rownbr != rownbr:
                        print('_rownbr ' + str(_rownbr))
                        print('rownbr ' + str(rownbr))
                        # should never happen...
                    assert(_rownbr == rownbr)

                    annot_idx = hlines_dict[self.time_series[rownbr]]['__annot_idx']
                    self.hlines_annotation_remove[annot_idx] = \
                        hlines_details_dict['visible_rows']
                    # ^ count down frames until it should be removed visually
            
            for ax_idx, ax in enumerate(self.axarr):
                scat_points = []
                for j in range(len(self.annotation_points)):
                    if self.annotation_points_visible[j][0] == True:
                        if self.annotation_points_visible[j][2] == ax_idx:
                            scat_points.append(self.annotation_points[j])
                if len(scat_points) > 0:
                    ax.scatter([point[0] for point in scat_points], [point[1] for point in scat_points])
            
            self.all_annotations_li = self.annotations_li + self.hlines_li + self.hlines_annotation_li
            return (*self.lines, *self.scatters, *self.contours, *self.all_annotations_li)
        return animate