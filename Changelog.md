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
