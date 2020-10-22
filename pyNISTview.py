#!/usr/local/anaconda3/bin/pythonw
"""Process .sempa files."""

import wx
from wx.lib.scrolledpanel import ScrolledPanel
from wx.lib.stattext import GenStaticText
from wx.adv import HyperlinkCtrl, HyperlinkEvent
# from wx.lib.imagebrowser import ImagePanel
from wxmplot import PlotPanel, ImagePanel

import glob
import os

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasAgg as FigureCanvas

from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter, median_filter

import pynistview_utils as pyn
import selelems as sel

import cv2

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
sigma = 2

# Specify non-local denoising strength h. Larger h -> more denoising.
h = 20

# Specify a scale for drawing the vector arrows. Smaller number -> longer
# arrow.
arrow_scale = 0.2

# Specify a color for the vector arrows.
# arrow_color = 'black'
arrow_color = 'white'

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
default_arrow_scale = 0.2

# Color for the vector arrows.
# default_arrow_color = 'black'
default_arrow_color = 'white'


class pyNISTView():
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

        self.raw_panel = RawViewPanel(self.main_panel)
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

        self.Bind(wx.EVT_MENU, self.on_open_files,
                  source=open_files_menu_item)

        self.Bind(wx.EVT_MENU, self.on_save,
                  source=save_menu_item)

        self.Bind(wx.EVT_MENU, self.on_export_with_scale,
                  source=export_with_scale_menu_item)

        self.Bind(wx.EVT_MENU, self.on_export,
                  source=export_without_scale_menu_item)

        self.Bind(wx.EVT_MENU, self.on_quit,
                  source=quit_menu_item)

        self.SetMenuBar(menu_bar)

    def create_toolbar(self):
        """Add a toolbar"""

        toolbar = self.CreateToolBar()
        open_tool = toolbar.AddTool(wx.ID_OPEN, 'Open', wx.Bitmap(
            'img/topen.png'))
        toolbar.AddSeparator()
        quit_tool = toolbar.AddTool(wx.ID_EXIT, 'Quit', wx.Bitmap(
            'img/texit.png'))
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
            self.show_raw_images()
            self.work_panel.set_model(self.model)
            self.set_status('Loaded')
        else:
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

    def show_raw_images(self):
        # Make life easier
        images = [self.model.m_1, self.model.m_2, self.model.intensity]

        # Reset the contents of raw_panel
        sizer = self.raw_panel.GetSizer()
        sizer.Clear(True)

        # Add the images
        for i in range(len(images)):
            image = images[i]
            name = os.path.basename(self.model.file_names[i])

            ip = ImagePanel(self.raw_panel, size=(
                image_min_width, image_min_height))
            ip.display(pyn.rescale(np.flip(image, axis=0), 0, 1))

            file_name = wx.StaticText(self.raw_panel, label=name)
            file_name.SetForegroundColour(wx.Colour(255, 255, 255))

            sizer.Add(ip, 1, 0, 0)
            sizer.Add(file_name, 1, wx.EXPAND, panel_spacing)

        # Refresh the sizer to redraw the screen
        sizer.Layout()

    def set_status(self, message):
        self.status_bar.SetStatusText(message)

# class RawImagePanel(wx.Panel):
#     def __init__(self, parent, image):
#         super().__init__(parent, style=wx.BORDER_SUNKEN)
#
#         self.figure = Figure()
#         self.axes = self.figure.add_subplot(111)
#         self.canvas = FigureCanvas(self.figure)
#         self.image = self.axes.imshow(image, aspect='auto')
#
#         self.sizer = wx.BoxSizer(wx.VERTICAL)
#         self.sizer.Add(self.canvas, 1, wx.ALIGN_CENTER
#                        | wx.EXPAND, panel_spacing)
#         self.SetSizer(self.sizer)
#         self.Fit()


class RawViewPanel(wx.Panel):
    """Define panel for raw views."""

    def __init__(self, parent):
        super().__init__(parent, style=wx.BORDER_SUNKEN)

        sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetBackgroundColour('#21242a')

        self.SetMinSize(wx.Size(image_min_width, image_min_height))
        self.SetSizer(sizer)


class WorkPanel(wx.Panel):
    """Define panel for doing work."""

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

        self.title_font = wx.SystemSettings.GetFont(wx.SYS_SYSTEM_FONT)
        self.title_font.SetPointSize(18)

        # self.default_text = GenStaticText(
        #     self, label='Please open a set of files', style=wx.ALIGN_CENTER)
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

        font = wx.SystemSettings.GetFont(wx.SYS_SYSTEM_FONT)
        font.SetPointSize(18)

        label = wx.StaticText(self, label=label)
        label.SetFont(font)
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

                ip = ImagePanel(self, style=wx.BG_STYLE_SYSTEM)
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

        # sizer = self.GetSizer()
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

        self.SetSizerAndFit(sizer)
        sizer.FitInside(self)

        # Refresh the sizer to redraw the screen
        sizer.Layout()


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
        self.m_1_delined = pyn.remove_line_errors(self.get_m_1(), self.lines)
        self.m_2_delined = pyn.remove_line_errors(self.get_m_2(), self.lines)

        self.m_1_delined = pyn.rescale(self.m_1_delined, self.get_m_1_min(),
                                       self.get_m_1_max())
        self.m_2_delined = pyn.rescale(self.m_2_delined, self.get_m_2_min(),
                                       self.get_m_2_max())

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
        axis_1, axis_2 = 'M{}'.format(
            self.get_axis_1()), 'M{}'.format(self.get_axis_2())

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

        self.m_1_offset, self.m_2_offset = offsets

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
        axis_1, axis_2 = 'M{}'.format(
            self.get_axis_1()), 'M{}'.format(self.get_axis_2())

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

    def show_images(self):
        # Let there be light

        # Add a sizer for variables: additional_offset
        variables_box = self.get_variables_box()

        # Add a sizer for images to show: fitted_denoised
        axis_1, axis_2 = 'M{}'.format(
            self.get_axis_1()), 'M{}'.format(self.get_axis_2())

        images = [self.m_1_denoised, self.m_2_denoised]

        labels = [axis_1 + ' denoised, offset {}'.format(self.m_1_offset),
                  axis_2 + ' denoised, offset {}'.format(self.m_2_offset)]

        images_box = self.get_images_box(images, labels)

        # Put it all together
        self.GetSizer().AddMany(
            [(variables_box, 0, 3 * wx.BOTTOM, panel_spacing),
             (images_box, 1, wx.EXPAND, panel_spacing)])

        # Refresh the sizer to redraw the screen
        # self.GetSizer().Fit(self)
        self.GetSizer().Layout()
        self.GetSizer().FitInside(self)

    def get_variables_box(self):
        font = wx.SystemSettings.GetFont(wx.SYS_SYSTEM_FONT)
        font.SetPointSize(18)

        scale_label = wx.StaticText(self, label=self.get_axis_1())
        scale_label.SetFont(font)
        scale_label.SetForegroundColour(wx.Colour(255, 255, 255))
        scale_input = wx.TextCtrl(self)
        scale_input.SetValue(str(self.arrow_scale))

        color_label = wx.StaticText(self, label=self.get_axis_2())
        color_label.SetFont(font)
        color_label.SetForegroundColour(wx.Colour(255, 255, 255))
        color_input = wx.TextCtrl(self)
        color_input.SetValue(self.arrow_color)

        self.variables_box = wx.BoxSizer(wx.HORIZONTAL)
        self.variables_box.AddMany(
            [(scale_label, 0, wx.RIGHT, panel_spacing),
             (scale_input, 0, wx.RIGHT, 4 * panel_spacing),
             (color_label, 0, wx.RIGHT, panel_spacing),
             (color_input, 0, 0, panel_spacing)])

        return self.variables_box


def main():
    app = wx.App()
    frame = pyNISTViewFrame()
    frame.Show()
    app.MainLoop()


if __name__ == '__main__':
    main()
