Place your portfolio images and videos in this folder.
Suggested names:
- profile.jpg
- prism-preview.png
- rootcause-demo.mp4
- edunote-prototype.jpg

Then reference them in templates/index.html like:
{{ url_for('static', filename='images/profile.jpg') }}
