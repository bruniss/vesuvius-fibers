![image](https://github.com/user-attachments/assets/ada642de-0372-45bd-9a5f-33d591c52312)


this repository is for fiber detection and prelimary autosegmentation methods for fibers in the herculaneum scrolls. there is a short writeup in here as well called fibers.pdf that gives some background information. the code in this repo is quite bad, and is filled with bugs, but i wanted to get it to the community faster than i saw myself refining this to a better state.

### to run fiber detection with the 3d model, 

you must first install nnunetv2 by following these instructions: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md

then you have to configure environment variables, more info is located here: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/set_environment_variables.md

this is typically all that is required on a linux machine, replacing the paths with your own.

```bash
export nnUNet_raw="/home/sean/wsl_docs/nnUNet/nnunet_data/nnUNet_raw"
export nnUNet_preprocessed="/home/sean/wsl_docs/nnUNet/nnunet_data/nnUNet_preprocessed"
export nnUNet_results="/home/sean/wsl_docs/nnUNet/nnunet_data/nnUNet_results"
```

then do the following:
 - download the checkpoint_final.pth from here https://dl.ash2txt.org/community-uploads/bruniss/Fiber%20Following/Model/nnUNet_results/Dataset004_fiber3/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/ and place it in your local folder (pay attention to this full path, as it is important. for me the local path would be `/home/sean/wsl_docs/nnUNet/nnunet_data/nnUNet_results/Dataset004_fiber3/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth` 
 - place the 3d tif you wish to infer on in the infer_in directory, ensuring you add the _0000 suffix to the end (this is a modality indicator, which i wont go into detail here. nnunets page has more info)
 - type `nnUNetv2_predict -i /home/sean/wsl_docs/nnUNet/infer -o /home/sean/wsl_docs/nnUNet/infer_out -d 004 -c 3d_fullres -f 0` and hit enter.
 - your results should infer here. if you run into issues please reach out or refer to nnunets documentation

### to test preliminary and very buggy meshing or fiber detection methods

i have uploaded some larger blocks that i ran inference and computed connected components for already here: https://dl.ash2txt.org/community-uploads/bruniss/Fiber%20Following/Fiber%20Predictions/
- 07000-11249_batch250/  # a 4250 sized block broken into 250 tif stacks    
- 7000_11249_originals.zarr/ # the original volume from this large block
- 7000_11249_predictions.zarr/ # the raw predictions from this block. 1 valued pixels are horizontals, 2 are verticals
- hz-cc-6.zarr/ # the horizontal fiber connected components , with 6 connectivity
- vt-cc-6.zarr/ # the vertical connected components , with 6 connectivity

this block is from 7000.tif to 11249.tif and from x1961:5393, y2135:5280

i also have some smaller ones located here https://drive.google.com/drive/u/0/folders/15a9nGFzF9NvP718zJ9q80nMGAP7MHkfK

for these methods i would recommend starting with the smaller ones, i did not yet optimize these for such large arrays and you're likely to run out of memory


### for the vertical fiber following method, 
- either load a precomputed connected components array, or load a raw prediction volume (for example stacked_1000_crop_predictions.tif from the drive above)
- if starting from raw predicitons, extract the class from the volume using the button and choosing the right value
- compute connected components with run cc3d
- select the vertical fibers you wish to use (tip: use the dropper tool), and add them to a comma separatead list in the proper order, the order is important here
- run compute nodes from labels with this list
- run connect splines from nodes with the same list
- to export a mesh obj, you can open the napari console and enter the following function definition ( i am really meaning to add this i just have been too busy to work on this at all the last week )
    ```python
    def quick_export_splines_to_obj(layer_name, filename='quick_export_splines.obj'):
        layer = viewer.layers[layer_name]
        splines = layer.data
        if isinstance(splines, np.ndarray) and splines.ndim == 2:
            splines = [splines]
        
        with open(filename, 'w') as f:
            vertex_count = 1
            for spline in splines:
                for point in spline:
                    f.write(f"v {point[2]} {point[1]} {point[0]}\n")
                f.write("l")
                for i in range(len(spline)):
                    f.write(f" {vertex_count + i}")
                f.write("\n")
                vertex_count += len(spline)
        print(f"Splines exported to {filename}")
    ```
- to generate the obj then type `quick_export_splines_to_obj('interp splines from nodes')`
- once you export this obj, you will need to create faces and such for it. to do this you can load the mesh up in meshlab, apply ball pivoting reconstruction, and then i would recommend using a laplacian smoothing filter. then check for non-manifold edges and you should have a workable mesh which you can import into khartes to inspect

### for the horizontal/vertical matching method
- repeat the same steps as above to the point where you have two volumes of connected components, one for verticals and one for horizontals
- select a starting vertical from the array (again i recommend using the dropper and zooming/panning around to find a good starter)
- select the the proper source labels for the vertical components and use the horizontals as the target, and insert the fiber id, select vertical , and direction of inside. i use max steps of about 15, but you can play with this to see if you like a different one
- hit find fibers. a list of all the horizontal fibers that presumably intersect with this vertical will be added to layers
- now do the opposite of this process, but from the found labels as the source, and verticals as the target -- you are now casting "backwards" to try and find horizontals for a new vertical, so direction will be "outside" and fiber type will be horizontal
- repeat this process for as long as you like, i currently have yet to implement a way to mesh this output. i think this method has some promise


the 3d model is generated using nnunetv2 (https://github.com/MIC-DKFZ/nnUNet/tree/master), from the following authors: 
Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring 
method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211.

labels were created using the software dragonfly3d , located here: https://www.theobjects.com/index.html
