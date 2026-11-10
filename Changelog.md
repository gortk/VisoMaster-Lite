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
- Optimization on the restorer/advanced: stability, speed , swap preservation (donor face)
										
 

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
