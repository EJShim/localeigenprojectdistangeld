import sys
import torch
import numpy as np
from pathlib import Path
import utils
from model_manager import get_model_manager
import vtk
from vtk.util import numpy_support
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from superqt import QDoubleSlider


if not torch.cuda.is_available():
    device = torch.device("cpu")
    print("GPU not available, running on CPU")
else:
    device = torch.device("cuda")


class LatentApp(QMainWindow):
    def __init__(self):
        super().__init__()

        parts = [
            "eyes","ears","temporal",
            "neck", "back", "mouth",
            "chin", "cheeks", "cheekbones",
            "forehead", "jaw", "nose"
        ]

        self.setCentralWidget(QWidget())
        self.resize(3800, 2000)

        self.root_layout = QHBoxLayout()
        self.centralWidget().setLayout(self.root_layout)

        self.renWidget = [
            QVTKRenderWindowInteractor(),
            QVTKRenderWindowInteractor()
        ]
        self.rens = []

        for renWidget in self.renWidget:
            renWidget.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
            renWin = renWidget.GetRenderWindow()
            renderer = vtk.vtkRenderer()
            renderer.SetBackground(1,1,1)
            renWin.AddRenderer(renderer)
            self.rens.append(renderer)
            self.root_layout.addWidget(renWidget, 2)
            renWin.Render()
        self.renWidget[0].AddObserver("InteractionEvent", lambda i, e : self.renWidget[1].GetRenderWindow().Render())
        self.renWidget[1].AddObserver("InteractionEvent", lambda i, e : self.renWidget[0].GetRenderWindow().Render())


        # Initialize Manager
        config_fn = Path("demo_files/config.yaml")
        configurations = utils.get_config(config_fn)


        self.manager = get_model_manager(
            configurations=configurations,
            device=device,
            precomputed_storage_path=Path("demo_files")
        )
        self.manager.resume(Path("demo_files/checkpoints"))
        self.normalization_dict = torch.load(Path("demo_files/norm.pt"))

        
        # Initialize Latent Vector
        self.z = torch.randn([1, self.manager.model_latent_size])

        # Initialize Slider Widget
        slider_widget = QWidget()
        self.root_layout.addWidget(slider_widget, 1)
        self.slider_layout = QVBoxLayout()
        slider_widget.setLayout(self.slider_layout)
        self.sliders = []

        for idx, val in enumerate( self.z[0] ):
            if idx % 5 == 0:
                label = QLabel(parts[ int(idx / 5)])
                self.slider_layout.addWidget(label)
            slider = QDoubleSlider(Qt.Orientation.Horizontal)
            slider.setRange(-2.5, 2.5)
            slider.setValue( val )
            slider.valueChanged.connect(lambda val, idx=idx : self.on_slider(val, idx))
            self.slider_layout.addWidget(slider)
            self.sliders.append(slider)
        
        # Update slider widget
        self.update_slider()


        # Initialize Mesh
        self.tmp_polydata = read_ply("configurations/uhm_template.ply")        
        self.tmp_polydata.GetPointData().RemoveArray("RGBA")        
        self.tmp_data = numpy_support.vtk_to_numpy(self.tmp_polydata.GetPoints().GetData())
        tmp_actor = make_actor(self.tmp_polydata)
        tmp_actor.GetMapper().SetScalarRange(0, 0.1)
        
        self.rens[0].AddActor(tmp_actor)
        self.rens[0].ResetCamera()

        # Initialize Prediction Mesh
        self.pred_polydata = vtk.vtkPolyData()
        self.pred_polydata.DeepCopy(self.tmp_polydata)
        self.pred_data = numpy_support.vtk_to_numpy(self.pred_polydata.GetPoints().GetData())
        pred_actor = make_actor(self.pred_polydata)
        self.rens[1].AddActor(pred_actor)
        self.rens[1].SetActiveCamera( self.rens[0].GetActiveCamera() )

        # Initial Prediction
        init_pred = self.predict()
        self.pred_data[:] = init_pred
        self.pred_polydata.GetPoints().Modified()

        self.tmp_data[:] = init_pred
        self.tmp_polydata.GetPoints().Modified()
        
        self.color_array = np.zeros(self.tmp_polydata.GetNumberOfPoints())
        color_data = numpy_support.numpy_to_vtk(self.color_array)
        color_data.SetName("dist")
        self.tmp_polydata.GetPointData().SetScalars(color_data)


        # Update Distance
        self.update_distance()

        self.redraw()
    
    def redraw(self):
        for widget in self.renWidget:
            widget.GetRenderWindow().Render()

    def update_distance(self):
        dist = np.linalg.norm(self.pred_data - self.tmp_data, axis=1)

        self.color_array[:] = dist
        self.tmp_polydata.GetPointData().GetScalars().Modified()

    def on_slider(self, val, idx):

        self.z[0][idx] = val
        pred_v = self.predict()
        self.pred_data[:] = pred_v
        self.pred_polydata.GetPoints().Modified()


        self.update_distance()
        self.redraw()
    def update_slider(self):
        for i, slider in enumerate(self.sliders):
            slider.setValue(self.z[0][i])

    def predict(self):
        gen_verts = self.manager.generate(self.z.to(device))[0, :, :]
        gen_verts = gen_verts * self.normalization_dict['std'].to(device) + self.normalization_dict['mean'].to(device)
        gen_verts = gen_verts.detach().cpu().numpy()

        return gen_verts





    def closeEvent(self, QCloseEvent):        
        super().closeEvent(QCloseEvent)
        for widget in self.renWidget:
            widget.Finalize()



def read_ply(filepath):
    reader = vtk.vtkPLYReader()
    reader.SetFileName(filepath)
    reader.Update()
    return reader.GetOutput()

def make_actor(polydata):
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    return actor

if __name__ == "__main__":

    app = QApplication(sys.argv)
    window = LatentApp()
    window.show()
    sys.exit(app.exec_())


    exit()

    # Initialize Renderer
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    renWin = vtk.vtkRenderWindow()
    renWin.SetSize(1000, 1000)
    iren.SetRenderWindow(renWin)
    ren = vtk.vtkRenderer()
    renWin.AddRenderer(ren)




    
    config_fn = Path("demo_files/config.yaml")
    configurations = utils.get_config(config_fn)


    manager = get_model_manager(
        configurations=configurations,
        device=device,
        precomputed_storage_path=Path("demo_files")
    )
    manager.resume(Path("demo_files/checkpoints"))

    normalization_dict = torch.load(Path("demo_files/norm.pt")) 
    print(normalization_dict["mean"].shape, normalization_dict["std"].shape)

    # read template
    tmp_polydata = read_ply("configurations/uhm_template.ply")
    
    point_array = numpy_support.vtk_to_numpy(tmp_polydata.GetPoints().GetData())
    tmp_actor = make_actor(tmp_polydata)
    ren.AddActor(tmp_actor)

    # Inference
    print(manager.model_latent_size)

    z = torch.randn([1, manager.model_latent_size])
    gen_verts = manager.generate(z.to(device))[0, :, :]
    gen_verts = gen_verts * normalization_dict['std'].to(device) + normalization_dict['mean'].to(device)
    gen_verts = gen_verts.detach().cpu().numpy()
    pred_polydata = vtk.vtkPolyData()    
    pred_polydata.DeepCopy(tmp_polydata)
    pred_polydata.GetPointData().RemoveArray("RGBA")
    
    pred_polydata.GetPoints().SetData( numpy_support.numpy_to_vtk(gen_verts) )
    pred_actor = make_actor(pred_polydata)
    pred_actor.SetPosition(2.5, 0, 0)
    ren.AddActor(pred_actor)





    ren.ResetCamera()

    renWin.Render()
    iren.Start()
