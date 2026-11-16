### 1.0.9

- Added new feature: tags for embeddings. Now you can right click on an embedding file and then "Tag", this will
  open a pop up where you can put persistent tags to your embeddings. The search function for tags now also searches
  tags so you can now have easy access to certain type of embeddings when you need them e.g "brown eyes" will bring 
  up the embeddings of people who have brown eyes.
  
- Separated embeddings from the last_workspace file to a separate "last_workspace_embeddings.jsonl" file that will now
  contain your embeddings, so sharing the "last_workspace" file should not be a risk of privacy. ⚠ Always check your file
  before sharing. In cases with many embeddings loaded (my case) it was bloating the last_workspace file by 5 MB. Now 
  it weights around 32 kb.
  
- Lazy loading of embedding tabs; now the app loads all your tabs but doesn't really load the embeddings inside them until
  you activate the tab. This should speed up a little more the startup times if you auto-load your workspace. 

### 1.0.8

- Added a popup window when generating tensorrt engines (the tensor cache)

### 1.0.7

- Added a "Temporal Color Smoothing" toggle under "Textures and colors" in "Face Swap" tab to configure it
  and turn it OFF if needed. It only runs for videos, not stills. Hopefully this fixes "solarized" swaps in
  certain bright videos.
- Fix for texture transfer in MODE 2 which solarized the videos if activated - Thanks @Alex1969 for reporting
  
### 1.0.6

- Fixed "Edit Faces" toggle not unloading its models when turning it OFF.
- Added new "Camera Gaze Lock" for recast mode, this saves you from having to turn on face editor
  just to fix badly detected pupils. Don't expect this to make your videos magically always gaze
  to the camera, as its more like a manual fix when required.
- Better, more precise seek bar, with frame ruler that adapts to zoom controls.
- Fixed the seek bar head that was conflicting with the segments so dragging it may cause segments to
  be unwillingly interacted with.
- Better UI rearrangement 

### 1.0.5

- Fixed the annoying toast "Cannot auto-load target media while an issue scan is running" when the
  issue scanner is running.
- Fixed issue where the "scan for issues" would mark "bad frames" but when the user stands over those 
  frames, the swap is still done correctly. Now those are not detected and the output video gets the 
  full swap instead of dropping those frames.  
- Even more patching done to prevent last_workspace.json from getting corrupted. I'm afraid this can 
  be a repeated theme as the issue seems rather random at the moment, so its difficult to test.
- Enabling the "auto mouth expression" no longer freezes the app. 
- New: no more markers. Now the user has a mouse dragging mechanic to create segments.
								
### 1.0.4

- Hopefully fixed the bug where sometimes the last_workspace.json gets corrupted and replaced with a defaults copy.
- Added the first 'official' extension: inpainting.
- Fixed the long (unnecessary) menu that appeared in the profiles in face editor.

### 1.0.3

- A lot of code refactor to try to separate ui threads
- Fixed "drop files here..." appearing in the target pane after files are loaded in it
- Added a new "Loading..." transparent watermark on the preview when the app is loading models.
- Added a simple splash screen when the app is loading.
- Hopefuly Fix the K/V map that survives face removal.
  current_kv_tensors_map on the main window can still point at the removed face’s assigned_kv_map, so the tensors stay on the GPU. (memory leak)

### 1.0.2

- Added a basic extensions support. In the new 'extensions' folder you will find an example extension
  open it to read more info on extensions. Change its name removing the leading "_" and it will become
  active.

### 1.0.1

- Re added target pane search: made a simple search like for the embeddings, no extra filtering.
- The app will no longer try to "play" an image file if the user clicks it in the preview window.
- The app now saves the size of the thumbnails in the target pane.

### Initial release 1.0.0

Removed:
- Search in the target pane
- Screenshots (we simply use the 'save image' for that)
- Hair Matting model (never seen any changes for me despite loading the model)
- Webcam usage.
- 'Glass Studio' theme
- Theater mode
- Fullscreen mode
- Jobs (still can do batches but no programmed jobs)
- MCP integration
- No more contextual menu over the video (right click menu). Right click + hold now pans over the media.

Added:
- Merged the last pull request from (@Masterofobzene) with a fix for the panes not being resizable
- Target pane now dynamically resizes the thumbnail previews with ctrl + wheel (@Masterofobzene)
- Small dedicated new icons for single-frame steps from @Smakaho
