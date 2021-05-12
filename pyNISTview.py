#!/usr/local/anaconda3/bin/pythonw
"""Process .sempa files."""

import wx
from wx.lib.scrolledpanel import ScrolledPanel
from wx.adv import HyperlinkCtrl

from wxmplot import ImagePanel

import glob, os

import numpy as np
from matplotlib.figure import Figure
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas

from scipy.ndimage import median_filter

import pynistview_utils as pyn

# import pynistview_utils as pyn

# Define some defaults for the UI
app_name = 'pyNISTView'
raw_label = 'Raw Images'
processed_label = 'Processed Images'
screen_scale = 3/4
button_spacing = 8
panel_spacing = 8
image_min_width, image_min_height = 256, 256

# Specify the number of features - 1 in the image. This is used to segment
# and create masks.
segments = 50

# Specify gaussian denoising sigma
sigma = 5

# Specify non-local denoising strength h. Larger h -> more denoising.
h = 20

# Specify a scale for drawing the vector arrows. Smaller number -> longer
# arrow.
arrow_scale = 2

# Specify a color for the vector arrows.
arrow_color = 'black'
# arrow_color = 'white'

sempa_file_suffix = 'sempa'

# For GradientPanel
#
# Lines for averaging for basic denoising
default_lines = 20

# Number of features
default_segments = 2

# For DenoisePanel
#
# Gaussian denoising sigma
default_sigma = 2

# Non-local denoising strength. Larger h -> more denoising
default_h = 20

# For ResultsPanel
#
# Scale for drawing the vector arrows. Smaller number -> longer arrow.
default_arrow_scale = 2

# Color for the vector arrows.
# default_arrow_color = 'black'
default_arrow_color = 'white'


class pyNISTView:
    """Main class for processing SEMPA images"""

    def __init__(self, file_path):
        file = file_path[:file_path.rfind('_')]

        self.init_files(file)

    def init_files(self, file):

        files = np.asarray(
            glob.glob('{}*x*{}'.format(file, sempa_file_suffix)))
        files = np.append(files, glob.glob(
            '{}*y*{}'.format(file, sempa_file_suffix)))
        files = sorted(np.append(files, glob.glob(
            '{}*z*{}'.format(file, sempa_file_suffix))))

        self.dirname = os.path.basename(os.path.dirname(files[0]))
        self.basename = os.path.basename(files[0])

        self.intensity, _ = pyn.image_data(files[0])
        self.m_1, self.axis_1 = pyn.image_data(files[2])
        self.m_2, self.axis_2 = pyn.image_data(files[3])

        self.file_names = [files[2], files[3], files[0]]

        # Get file dimensions

        self.m_1_ydim, self.m_1_xdim = self.m_1.shape
        self.m_2_ydim, self.m_2_xdim = self.m_2.shape

        # Extract extrema for later processing

        self.m_1_min, self.m_1_max = self.m_1.min(), self.m_1.max()
        self.m_2_min, self.m_2_max = self.m_2.min(), self.m_2.max()

        # Flatten the intensity image

        self.intensity_blurred = median_filter(self.intensity, 3)
        self.intensity_flat = self.intensity - self.intensity_blurred

        self.scale = pyn.get_scale(files[0])

    def get_name(self):
        return self.dirname + '/' + self.basename[:self.basename.rfind('_')]


class pyNISTViewFrame(wx.Frame):
    """Define main frame."""

    def __init__(self):
        screen_dims = np.asarray(wx.GetDisplaySize())
        window_dims = screen_dims * screen_scale
        super().__init__(parent=None, title=app_name, size=window_dims)

        self.title_font = wx.SystemSettings.GetFont(wx.SYS_SYSTEM_FONT)
        self.title_font.SetPointSize(18)

        self.Center()

        self.model = None

        self.create_menu()
        self.create_toolbar()
        self.status_bar = self.CreateStatusBar()
        self.create_panels()

    def create_panels(self):
        # Create the main panel
        self.main_panel = wx.Panel(self)
        self.main_panel.SetBackgroundColour('gray')

        # Create the raw and working panels
        raw_title = wx.StaticText(
            self.main_panel, label=raw_label, style=wx.ALIGN_CENTER)
        raw_title.SetFont(self.title_font)
        processed_title = wx.StaticText(
            self.main_panel, label=processed_label, style=wx.ALIGN_CENTER)
        processed_title.SetFont(self.title_font)

        self.raw_panel = RawPanel(self.main_panel)
        self.work_panel = WorkPanel(self.main_panel, self.model)
        # self.button_panel = ButtonPanel(self.main_panel)

        # Put the panels in a flex sizer
        flex_sizer = wx.FlexGridSizer(
            rows=2, cols=2, vgap=panel_spacing, hgap=panel_spacing)

        flex_sizer.AddMany(
            [(raw_title, 1, wx.EXPAND, panel_spacing),
             (processed_title, 1, wx.EXPAND, panel_spacing),
             # (self.button_panel, 1, wx.EXPAND, panel_spacing),
             (self.raw_panel, 1, wx.EXPAND, panel_spacing),
             (self.work_panel, 1, wx.EXPAND, panel_spacing)])

        flex_sizer.AddGrowableRow(1, 1)
        flex_sizer.AddGrowableCol(1, 1)

        box_sizer = wx.BoxSizer(wx.HORIZONTAL)
        box_sizer.Add(flex_sizer, proportion=1,
                      flag=wx.ALL | wx.EXPAND, border=panel_spacing)

        self.main_panel.SetSizer(box_sizer)

    def create_menu(self):
        """Add a menu."""

        menu_bar = wx.MenuBar()

        # File menu
        file_menu = wx.Menu()

        open_files_menu_item = file_menu.Append(
            wx.ID_OPEN, '&Open...\tCtrl+O')
        save_menu_item = file_menu.Append(wx.ID_SAVE, '&Save\tCtrl+S')

        export_menu_item = wx.Menu()
        export_with_scale_menu_item = export_menu_item.Append(
            wx.ID_ANY, 'Export with scale bars',
            'Add scale bars to exported image')
        export_without_scale_menu_item = export_menu_item.Append(
            wx.ID_ANY, 'Export without scale bars', 'Export image')
        file_menu.AppendSubMenu(export_menu_item, '&Export...')

        file_menu.AppendSeparator()

        quit_menu_item = file_menu.Append(wx.ID_ANY, '&Quit\tCtrl+Q', 'Quit')

        # Edit menu
        edit_menu = wx.Menu()

        # View menu
        view_menu = wx.Menu()

        # Window menu
        window_menu = wx.Menu()

        # Help menu
        help_menu = wx.Menu()

        menu_bar.Append(file_menu, '&File')
        menu_bar.Append(edit_menu, '&Edit')
        menu_bar.Append(view_menu, '&View')
        menu_bar.Append(window_menu, '&Window')
        menu_bar.Append(help_menu, '&Help')

        self.Bind(wx.EVT_MENU, self.on_open_files, open_files_menu_item)

        self.Bind(wx.EVT_MENU, self.on_save, save_menu_item)

        self.Bind(wx.EVT_MENU, self.on_export_with_scale,
                  export_with_scale_menu_item)

        self.Bind(wx.EVT_MENU, self.on_export,
                  export_without_scale_menu_item)

        self.Bind(wx.EVT_MENU, self.on_quit, quit_menu_item)

        self.SetMenuBar(menu_bar)

    def create_toolbar(self):
        """Add a toolbar"""

        toolbar = self.CreateToolBar()
        open_tool = toolbar.AddTool(wx.ID_OPEN, 'Open', wx.Bitmap(
            'img/topen.png'), shortHelp='Open')

        toolbar.AddSeparator()

        quit_tool = toolbar.AddTool(wx.ID_EXIT, 'Quit', wx.Bitmap(
            'img/texit.png'), shortHelp='Quit')

        toolbar.Realize()

        self.Bind(wx.EVT_TOOL, self.on_open_files, open_tool)
        self.Bind(wx.EVT_TOOL, self.on_quit, quit_tool)

    def on_open_files(self, event):
        """Respond to open."""

        self.set_status('Working...')

        title = 'Choose a SEMPA file:'
        wildcard = 'SEMPA files(*.sempa)|*.sempa'
        dlg = wx.FileDialog(self, title, wildcard=wildcard,
                            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        if dlg.ShowModal() == wx.ID_OK:
            self.model = pyNISTView(dlg.GetPath())
        dlg.Destroy()

        if self.model is not None:
            self.raw_panel.set_model(self.model)
            self.work_panel.set_model(self.model)
            self.set_title(app_name + ': ' + self.model.get_name())
            self.set_status('Loaded')
        else:
            self.set_title(app_name)
            self.set_status('Nothing loaded')

    def on_save(self, event):
        """Respond to save"""

        print('In on_save().')

    def on_export_with_scale(self, event):
        """Respond to export with scale"""

        print('In on_export_with_scale().')

    def on_export(self, event):
        """Respond to export"""

        print('In on_export().')

    def on_quit(self, event):
        """Quit"""
        self.Close()

    def set_title(self, title):
        self.SetTitle(title)

    def set_status(self, message):
        self.status_bar.SetStatusText(message)


class RawPanel(ScrolledPanel):
    """Panel for raw images."""

    def __init__(self, parent):
        super().__init__(parent, style=wx.BORDER_SUNKEN)

        sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetBackgroundColour('#21242a')

        self.SetMinSize(wx.Size(image_min_width + 20, image_min_height))
        self.SetSizer(sizer)

        self.SetupScrolling()
        self.SetAutoLayout(True)

    def set_model(self, model):
        # Images to display
        images = [model.m_1, model.m_2, model.intensity]

        # Reset the contents
        sizer = self.GetSizer()
        sizer.Clear(True)

        dpi = wx.GetDisplayPPI().GetWidth()

        # Add the images
        for i in range(len(images)):
            image = images[i]
            name = os.path.basename(model.file_names[i])

            ip = ImagePanel(self, size=(
                image_min_width, image_min_height), dpi=dpi)
            ip.display(pyn.rescale(np.flip(image, axis=0), 0, 1))

            file_name = wx.StaticText(self, label=name)
            file_name.SetForegroundColour(wx.Colour(255, 255, 255))

            sizer.Add(ip, 1, 0, 0)
            # sizer.Add(canvas, 1, wx.EXPAND, 0)
            sizer.Add(file_name, 0, wx.EXPAND | wx.BOTTOM, panel_spacing)
            sizer.AddSpacer(2 * panel_spacing)

        # Refresh the sizer to redraw the screen
        sizer.AddStretchSpacer()
        sizer.FitInside(self)
        sizer.Layout()


class WorkPanel(wx.Panel):
    """Panel for doing work."""

    def __init__(self, parent, model):
        super().__init__(parent, style=wx.BORDER_SUNKEN)
        self.SetBackgroundColour('#272c34')

        self.nb = WorkNotebook(self, model)
        fit_page = FitPanel(self.nb)
        denoise_page = DenoisePanel(self.nb)
        offsets_page = OffsetsPanel(self.nb)
        results_page = ResultsPanel(self.nb)

        self.nb.AddPage(fit_page, "Fit")
        self.nb.AddPage(denoise_page, "Denoise")
        self.nb.AddPage(offsets_page, "Offsets")
        self.nb.AddPage(results_page, "Results")

        # Set to the results tab
        self.nb.SetSelection(self.nb.GetPageCount() - 1)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.nb, 1, wx.EXPAND)
        self.SetSizer(sizer)

    def set_model(self, model):
        self.nb.set_model(model)


class WorkNotebook(wx.Notebook):
    """Convenience class for managing model access"""

    def __init__(self, parent, model):
        super().__init__(parent)
        self.set_model(model)

    def get_model(self):
        return self.model

    def set_model(self, model):
        self.model = model
        self.update_model()

    def update_model(self):
        # Tell the pages that the model changed
        for i in range(self.GetPageCount()):
            self.GetPage(i).model_updated()

    def get_m_1_fitted(self):
        return self.m_1_fitted

    def set_m_1_fitted(self, value):
        self.m_1_fitted = value

    def get_m_2_fitted(self):
        return self.m_2_fitted

    def set_m_2_fitted(self, value):
        self.m_2_fitted = value

    def get_m_1_fitted_denoised(self):
        return self.m_1_fitted_denoised

    def set_m_1_fitted_denoised(self, value):
        self.m_1_fitted_denoised = value

    def get_m_2_fitted_denoised(self):
        return self.m_2_fitted_denoised

    def set_m_2_fitted_denoised(self, value):
        self.m_2_fitted_denoised = value

    def get_m_1_denoised(self):
        return self.m_1_denoised

    def set_m_1_denoised(self, value):
        self.m_1_denoised = value

    def get_m_2_denoised(self):
        return self.m_2_denoised

    def set_m_2_denoised(self, value):
        self.m_2_denoised = value

    def get_m_1_offset(self):
        return self.m_1_offset

    def set_m_1_offset(self, value):
        self.m_1_offset = value

    def get_m_2_offset(self):
        return self.m_2_offset

    def set_m_2_offset(self, value):
        self.m_2_offset = value

    def get_phases(self):
        return self.phases

    def set_phases(self, value):
        self.phases = value

    def get_magnitudes(self):
        return self.magnitudes

    def set_magnitudes(self, value):
        self.magnitudes = value


class PanelWithModel(ScrolledPanel):
    """Convenience abstract class for managing model access"""

    def __init__(self, parent):
        super().__init__(parent)

        self.variables_box = None
        self.images_box = None

        self.title_font = self.get_frame().title_font

        default_text = HyperlinkCtrl(
            self, label='Please open a set of files')
        default_text.SetFont(self.title_font)
        default_text.SetForegroundColour(wx.Colour(255, 255, 255))

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.AddStretchSpacer()
        sizer.Add(default_text, 0, wx.EXPAND, panel_spacing)
        sizer.AddStretchSpacer()

        self.SetupScrolling()
        self.SetAutoLayout(True)

        self.SetSizer(sizer)

        default_text.Bind(wx.adv.EVT_HYPERLINK, self.on_open_files)

    # Model getters and setters

    def get_model(self):
        return self.GetParent().get_model()

    def get_m_1(self):
        return self.get_model().m_1

    def get_m_1_min(self):
        return self.get_model().m_1_min

    def get_m_1_max(self):
        return self.get_model().m_1_max

    def get_m_2(self):
        return self.get_model().m_2

    def get_m_2_min(self):
        return self.get_model().m_2_min

    def get_m_2_max(self):
        return self.get_model().m_2_max

    def get_axis_1(self):
        return self.get_model().axis_1

    def get_axis_2(self):
        return self.get_model().axis_2

    def get_frame(self):
        return self.GetGrandParent().GetGrandParent()

    def set_status(self, message):
        self.get_frame().set_status(message)

    def clear_status(self):
        self.set_status('')

    def on_open_files(self, evt):
        # Need CallAfter to avoid a segmentation fault: the called
        # method kills off the widget handling this event
        wx.CallAfter(self.get_frame().on_open_files, evt)

    def update_model(self):
        self.GetParent().update_model()

    def model_updated(self):
        # Clear the sizer. Extend this to add more.
        self.GetSizer().Clear(True)

    def get_axes(self):
        return 'M{}'.format(self.get_axis_1()), 'M{}'.format(self.get_axis_2())

    def get_label_text(self, label):
        """Convenience wrapper for setting up labels"""

        label = wx.StaticText(self, label=label)
        label.SetFont(self.title_font)
        label.SetForegroundColour(wx.Colour(255, 255, 255))

        return label

    def get_variables_box(self):
        variables = self.get_variables()

        self.variables_box = wx.BoxSizer(wx.HORIZONTAL)

        for var in variables:
            self.variables_box.AddMany(
                                     [(var[0], 0, wx.RIGHT, panel_spacing),
                                      (var[1], 0, 0, panel_spacing)])
            self.variables_box.AddSpacer(3 * panel_spacing)

        return self.variables_box

    def get_images_box(self, images, labels):
        self.images_box = wx.BoxSizer(wx.HORIZONTAL)

        # Monkey business to display the images as large as possible and fit
        dpi = wx.GetDisplayPPI().GetWidth()
        panel_width = self.GetSize().GetWidth()
        image_count = len(images) / 2

        # Width of panel minus scroll bar minus padding / # images
        max_image_size = (panel_width - 40 - image_count
                          * 2 * panel_spacing) / image_count

        # Add the images
        # Expect to show an image for m_1 and m_2 one on top of the other
        for i in range(0, len(images), 2):
            vbox = wx.BoxSizer(wx.VERTICAL)

            # Put the label below each image
            for j in range(2):
                index = i + j
                ibox = wx.BoxSizer(wx.VERTICAL)
                image = images[index]
                label = labels[index]
                image_height, image_width = image.shape
                image_size = min((2 * image_width, max_image_size))

                # ip = ImagePanel(self, size=(image_width, image_height),
                ip = ImagePanel(self, size=(image_size, image_size),
                                dpi=dpi)
                # Need to flip the image since the array does the y axis
                # opposite display
                ip.display(pyn.rescale(np.flip(image, axis=0), 0, 1),
                           style=wx.BG_STYLE_TRANSPARENT)

                name = wx.StaticText(self, label=label, style=wx.ALIGN_LEFT)
                name.SetForegroundColour(wx.Colour(255, 255, 255))

                ibox.AddMany(
                    [(ip, 1, wx.ALL, panel_spacing),
                     (name, 0, wx.EXPAND | wx.LEFT | wx.TOP | wx.BOTTOM,
                        panel_spacing)])

                vbox.Add(ibox, 0, wx.RIGHT, panel_spacing)
                vbox.AddSpacer(4 * panel_spacing)

            self.images_box.Add(vbox, 0, wx.RIGHT, panel_spacing)

        return self.images_box

    def show_images(self):
        # Let there be light

        sizer = wx.BoxSizer(wx.VERTICAL)

        # Add a sizer for variables
        variables_box = self.get_variables_box()

        # Add a sizer for images to show
        images, labels = self.get_images()
        images_box = self.get_images_box(images, labels)

        # Put it all together
        sizer.AddMany(
            [(variables_box, 0, wx.BOTTOM, panel_spacing),
             (images_box, 0, wx.BOTTOM, panel_spacing)])

        # Refresh the sizer to redraw the screen
        self.SetSizer(sizer)
        sizer.Layout()
        self.FitInside()

    def get_image_for_array(self, image_array):
        dpi = wx.GetDisplayPPI()

        image_height, image_width = image_array.shape

        figure = Figure(figsize=(image_width // dpi.GetWidth(),
                                 image_height // dpi.GetHeight()),
                        frameon=False)

        axes = figure.add_subplot(111)

        axes.imshow(image_array, cmap='gray')
        axes.grid(False)
        axes.get_xaxis().set_visible(False)
        axes.get_yaxis().set_visible(False)

        canvas = FigureCanvas(self, 0, figure)

        return canvas

    def get_variables(self):
        return ()

    def get_images(self):
        return ((), ())


class FitPanel(PanelWithModel):
    """Fit the input data to remove background gradient"""

    def __init__(self, parent):
        super().__init__(parent)

        self.lines = default_lines
        self.segments = default_segments

    def model_updated(self):
        super().model_updated()

        self.remove_line_errors()
        self.mean_shift()
        self.segment()
        self.fit()
        self.show_images()

    def remove_line_errors(self):
        # self.m_1_delined = pyn.remove_line_errors(self.get_m_1(), self.lines)
        # self.m_2_delined = pyn.remove_line_errors(self.get_m_2(), self.lines)
        #
        # self.m_1_delined = pyn.rescale_to(self.m_1_delined, self.get_m_1())
        # self.m_2_delined = pyn.rescale_to(self.m_2_delined, self.get_m_2())
        self.m_1_delined = self.get_m_1()
        self.m_2_delined = self.get_m_2()

    def mean_shift(self):
        self.m_1_shift = pyn.mean_shift_filter(self.m_1_delined)
        self.m_2_shift = pyn.mean_shift_filter(self.m_2_delined)

    def segment(self):
        self.m_1_thresh, self.m_1_seg, self.m_1_masks = pyn.segment_image(
            self.m_1_shift, segments=self.segments)
        self.m_2_thresh, self.m_2_seg, self.m_2_masks = pyn.segment_image(
            self.m_2_shift, segments=self.segments)

    def fit(self):
        # m_1 first
        self.m_1_light_mask, self.m_1_dark_mask = \
            self.m_1_masks[0, 0], self.m_1_masks[0, 1]

        self.m_1_light_fit_all, self.m_1_light_fit = pyn.fit_image(
            self.get_m_1(), self.m_1_light_mask)
        self.m_1_dark_fit_all, self.m_1_dark_fit = pyn.fit_image(
            self.get_m_1(), self.m_1_dark_mask)

        self.m_1_light_fitted, self.m_1_dark_fitted = self.get_m_1() - \
            self.m_1_light_fit_all, self.get_m_1() - self.m_1_dark_fit_all

        # and m_2
        self.m_2_light_mask, self.m_2_dark_mask = \
            self.m_2_masks[0, 0], self.m_2_masks[0, 1]

        self.m_2_light_fit_all, self.m_2_light_fit = pyn.fit_image(
            self.get_m_2(), self.m_2_light_mask)
        self.m_2_dark_fit_all, self.m_2_dark_fit = pyn.fit_image(
            self.get_m_2(), self.m_2_dark_mask)

        self.m_2_light_fitted, self.m_2_dark_fitted = self.get_m_2() - \
            self.m_2_light_fit_all, self.get_m_2() - self.m_2_dark_fit_all

        # Rescale both
        self.m_1_light_fitted = pyn.rescale_to(
            self.m_1_light_fitted, self.get_m_1())
        self.m_2_light_fitted = pyn.rescale_to(
            self.m_2_light_fitted, self.get_m_2())

        # Save them to the notebook for subsequent processing
        self.GetParent().set_m_1_fitted(self.m_1_light_fitted)
        self.GetParent().set_m_2_fitted(self.m_2_light_fitted)

    def get_variables(self):
        lines_label = self.get_label_text('Lines')
        self.lines_input = wx.TextCtrl(self)
        self.lines_input.SetValue(str(self.lines))

        segments_label = self.get_label_text('Segments')
        self.segments_input = wx.TextCtrl(self)
        self.segments_input.SetValue(str(self.segments))

        self.lines_input.Bind(wx.EVT_KILL_FOCUS, self.on_lines_change)
        self.segments_input.Bind(
            wx.EVT_KILL_FOCUS, self.on_segments_change)

        return ((lines_label, self.lines_input),
                (segments_label, self.segments_input))

    def get_images(self):
        axis_1, axis_2 = self.get_axes()

        images = [self.m_1_delined, self.m_2_delined, self.m_1_seg,
                  self.m_2_seg, self.m_1_light_fitted, self.m_2_light_fitted]

        labels = [axis_1 + ' delined', axis_2 + ' delined', axis_1
                  + ' contours', axis_2 + ' contours', axis_1 + ' fit',
                  axis_2 + ' fit']

        return images, labels

    def on_lines_change(self, evt):
        new_value = int(self.lines_input.GetValue())

        if new_value != self.lines:
            self.set_status('Updating lines...')
            self.lines = new_value
            self.update_model()
            self.set_status('Updated lines')

    def on_segments_change(self, evt):
        new_value = int(self.segments_input.GetValue())

        if new_value != self.segments:
            self.set_status('Updating segments...')
            self.segments = new_value
            self.update_model()
            self.set_status('Updated segments')


class DenoisePanel(PanelWithModel):
    """Remove noise from the data"""

    def __init__(self, parent):
        super().__init__(parent)

        self.sigma = default_sigma
        self.h = default_h

    def get_m_1_fitted(self):
        return self.GetParent().get_m_1_fitted()

    def get_m_2_fitted(self):
        return self.GetParent().get_m_2_fitted()

    def model_updated(self):
        super().model_updated()

        self.remove_noise()
        self.show_images()

    def remove_noise(self):
        self.m_1_fitted_denoised, self.m_1_fitted_denoised_blurred = \
            pyn.clean_image(self.get_m_1_fitted(), sigma=self.sigma, h=self.h)
        self.m_2_fitted_denoised, self.m_2_fitted_denoised_blurred = \
            pyn.clean_image(self.get_m_2_fitted(), sigma=self.sigma, h=self.h)

        self.GetParent().set_m_1_fitted_denoised(self.m_1_fitted_denoised)
        self.GetParent().set_m_2_fitted_denoised(self.m_2_fitted_denoised)

    def get_variables(self):
        sigma_label = self.get_label_text('Sigma ')
        self.sigma_input = wx.TextCtrl(self)
        self.sigma_input.SetValue(str(self.sigma))

        h_label = self.get_label_text('h')
        self.h_input = wx.TextCtrl(self)
        self.h_input.SetValue(str(self.h))

        self.sigma_input.Bind(wx.EVT_KILL_FOCUS, self.on_sigma_change)
        self.h_input.Bind(wx.EVT_KILL_FOCUS, self.on_h_change)

        return ((sigma_label, self.sigma_input), (h_label, self.h_input))

    def get_images(self):
        axis_1, axis_2 = self.get_axes()

        images = [self.m_1_fitted_denoised, self.m_2_fitted_denoised]

        labels = [axis_1 + ' denoised', axis_2 + ' denoised']

        return images, labels

    def on_sigma_change(self, evt):
        new_value = int(self.sigma_input.GetValue())

        if new_value != self.sigma_input:
            self.set_status('Updating sigma...')
            self.sigma = new_value
            self.update_model()
            self.set_status('Updated sigma')

    def on_h_change(self, evt):
        new_value = int(self.h_input.GetValue())

        if new_value != self.h_input:
            self.set_status('Updating h...')
            self.h = new_value
            self.update_model()
            self.set_status('Updated h')


class OffsetsPanel(PanelWithModel):
    """Find and remove offsets from the data"""

    def __init__(self, parent):
        super().__init__(parent)

    def get_m_1_fitted_denoised(self):
        return self.GetParent().get_m_1_fitted_denoised()

    def get_m_2_fitted_denoised(self):
        return self.GetParent().get_m_2_fitted_denoised()

    def model_updated(self):
        super().model_updated()

        self.find_offsets()
        self.apply_offsets()
        self.calculate_phases_and_magnitudes()
        self.show_images()

    def find_offsets(self):
        offsets, status, message = \
            pyn.find_offsets(self.get_m_1_fitted_denoised(),
                             self.get_m_2_fitted_denoised())

        self.m_1_offset, self.m_2_offset, _ = offsets
        self.GetParent().set_m_1_offset(self.m_1_offset)
        self.GetParent().set_m_2_offset(self.m_2_offset)

    def apply_offsets(self):
        self.m_1_denoised = self.get_m_1_fitted_denoised() - self.m_1_offset
        self.m_2_denoised = self.get_m_2_fitted_denoised() - self.m_2_offset

        self.GetParent().set_m_1_denoised(self.m_1_denoised)
        self.GetParent().set_m_2_denoised(self.m_2_denoised)

    def calculate_phases_and_magnitudes(self):
        self.phases = pyn.get_phases(self.m_1_denoised, self.m_2_denoised)
        self.magnitudes = pyn.get_magnitudes(
            self.m_1_denoised, self.m_2_denoised)
        self.magnitudes /= self.magnitudes.max()

        self.GetParent().set_phases(self.phases)
        self.GetParent().set_magnitudes(self.magnitudes)

    def get_variables(self):
        m_1_label = self.get_label_text(self.get_axis_1())
        self.m_1_input = wx.TextCtrl(self)
        self.m_1_input.SetValue('0')

        m_2_label = self.get_label_text(self.get_axis_2())
        self.m_2_input = wx.TextCtrl(self)
        self.m_2_input.SetValue('0')

        return ((m_1_label, self.m_1_input),
                (m_2_label, self.m_2_input))

    def get_images(self):
        axis_1, axis_2 = self.get_axes()

        images = [self.m_1_denoised, self.m_2_denoised]

        labels = [axis_1 + ' denoised, offset {}'.format(self.m_1_offset),
                  axis_2 + ' denoised, offset {}'.format(self.m_2_offset)]

        return images, labels


class ResultsPanel(PanelWithModel):
    """Display the results of processing"""

    def __init__(self, parent):
        super().__init__(parent)

        self.arrow_scale = default_arrow_scale
        self.arrow_color = default_arrow_color

        self.dpi = wx.GetDisplayPPI()

    def get_intensity_final(self):
        return self.get_model().intensity_flat

    def get_m_1_final(self):
        return self.GetParent().get_m_1_denoised()

    def get_m_2_final(self):
        return self.GetParent().get_m_2_denoised()

    def get_m_1_offset(self):
        return self.GetParent().get_m_1_offset()

    def get_m_2_offset(self):
        return self.GetParent().get_m_2_offset()

    def get_phases(self):
        return self.GetParent().get_phases()

    def get_magnitudes(self):
        return self.GetParent().get_magnitudes()

    def model_updated(self):
        super().model_updated()

        self.show_images()

    def show_images(self):
        sizer = wx.BoxSizer(wx.VERTICAL)

        # Add a sizer for variables
        variables_box = self.get_variables_box()
        contrast_box = self.get_images_box()

        sizer.AddMany([(variables_box, 0, wx.BOTTOM, panel_spacing),
                       (contrast_box, 0, wx.BOTTOM, panel_spacing)])

        # sizer.SetSizeHints(self)
        self.SetSizer(sizer)
        sizer.Layout()
        self.FitInside()

    def get_images_box(self):
        axis_1, axis_2 = self.get_axes()

        v_sizer = wx.BoxSizer(wx.VERTICAL)
        h_sizer_1 = wx.BoxSizer(wx.HORIZONTAL)

        # Main image first
        title = self.get_model().get_name()

        img = pyn.render_phases_and_magnitudes(self.get_phases()[0],
                                               self.get_magnitudes())

        image_height, image_width, _ = img.shape
        dpi = wx.GetDisplayPPI().GetWidth()

        figure = Figure(figsize=(2 * image_width // dpi,
                                 2 * image_height // dpi),
                        frameon=False, constrained_layout=True)

        axes = figure.add_subplot(1, 1, 1)

        axes.imshow(img)
        pyn.show_vector_plot(self.get_m_1_final(), self.get_m_2_final(),
                             ax=axes, color=self.arrow_color,
                             scale=self.arrow_scale)
        axes.add_artist(ScaleBar(self.get_model().scale, box_alpha=0.8))

        axes.grid(False)
        axes.get_xaxis().set_visible(False)
        axes.get_yaxis().set_visible(False)

        axes.set_title(
            'Domains in the {}-{} plane for {}'.format(
                axis_1, axis_2, title),
            fontdict={'fontsize': 12}, color='white')

        canvas = FigureCanvas(self, 0, figure)

        h_sizer_1.Add(canvas, 0, wx.BOTTOM, panel_spacing)
        h_sizer_1.AddStretchSpacer()

        # Now angle legend
        legend_sizer = wx.BoxSizer(wx.VERTICAL)
        legend_sizer.AddStretchSpacer()

        figure_2 = Figure(frameon=False, constrained_layout=True)
        axes_2 = figure_2.add_subplot(111, polar=True)
        pyn.show_phase_colors_circle(axes_2)
        axes_2.set_title('Magnetization angle', fontdict={'fontsize': 10},
                         color='white')
        canvas_2 = FigureCanvas(self, 0, figure_2)

        legend_sizer.Add(canvas_2, 0, 0, 0)
        legend_sizer.AddStretchSpacer()

        h_sizer_1.Add(legend_sizer, 0, wx.EXPAND | wx.BOTTOM | wx.RIGHT, panel_spacing)

        # More nonsense
        # ip = ImagePanel(self, size=(2 * image_width, 2 * image_height), dpi=dpi)
        # ip.display(np.flip(img / 255, axis=0))
        #
        # h_sizer_1.Add(ip, 1, 0, 0)

        v_sizer.Add(h_sizer_1, 1, wx.EXPAND | wx.ALL, panel_spacing)

        # Flattened images
        h_sizer_2 = wx.BoxSizer(wx.HORIZONTAL)

        # intensity_ip = ImagePanel(self, size=(image_width, image_height), dpi=dpi)
        # intensity_ip.display(pyn.rescale(np.flip(self.get_intensity_final(), axis=0), 0, 1))
        # m_1_ip = ImagePanel(self, size=(image_width, image_height), dpi=dpi)
        # m_1_ip.display(pyn.rescale(np.flip(self.get_m_1_final(), axis=0), 0, 1))
        # m_2_ip = ImagePanel(self, size=(image_width, image_height), dpi=dpi)
        # m_2_ip.display(pyn.rescale(np.flip(self.get_m_2_final(), axis=0), 0, 1))

        intensity_ip = self.image_panel_box_for(self.get_intensity_final(),
                                                'Intensity flattened')
        m_1_ip = self.image_panel_box_for(self.get_m_1_final(),
                                          '{} flattened'.format(axis_1))
        m_2_ip = self.image_panel_box_for(self.get_m_2_final(),
                                          '{} flattened'.format(axis_2))

        h_sizer_2.Add(intensity_ip, 1, 0, 0)
        h_sizer_2.AddStretchSpacer()
        h_sizer_2.Add(m_1_ip, 1, 0, 0)
        h_sizer_2.AddStretchSpacer()
        h_sizer_2.Add(m_2_ip, 1, 0, 0)

        v_sizer.AddSpacer(4 * panel_spacing)
        v_sizer.Add(h_sizer_2, 0, wx.ALL, panel_spacing)

        return v_sizer

    def image_panel_box_for(self, image, title):
        image_height, image_width = image.shape
        dpi = wx.GetDisplayPPI().GetWidth()

        v_box = wx.BoxSizer(wx.VERTICAL)

        ip = ImagePanel(self, size=(image_width, image_height), dpi=dpi)
        ip.display(pyn.rescale(np.flip(image, axis=0), 0, 1))
        title_text = wx.StaticText(self, label=title)
        title_text.SetForegroundColour(wx.Colour(255, 255, 255))

        v_box.AddMany([(ip, 0, wx.BOTTOM, 2 * panel_spacing), (title_text, 0,
                                                       wx.EXPAND, 0)])

        return v_box

    def get_variables(self):
        scale_label = self.get_label_text('Arrow scale')
        self.scale_input = wx.TextCtrl(self)
        self.scale_input.SetValue(str(self.arrow_scale))

        color_label = self.get_label_text('Arrow color')
        self.color_input = wx.TextCtrl(self)
        self.color_input.SetValue(self.arrow_color)

        return ((scale_label, self.scale_input),
                (color_label, self.color_input))

    def get_images(self):
        axis_1, axis_2 = self.get_axes()

        images = [self.get_m_1_final(), self.get_m_2_final()]

        labels = [axis_1 + ' denoised, offset {}'.format(self.get_m_1_offset()),
                  axis_2 + ' denoised, offset {}'.format(self.get_m_2_offset())]

        return images, labels


def main():
    app = wx.App()
    frame = pyNISTViewFrame()
    frame.Show()
    app.MainLoop()


if __name__ == '__main__':
    main()
