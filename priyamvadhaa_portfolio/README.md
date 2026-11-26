# Priyamvadhaa Hoskere — Portfolio

A responsive personal portfolio built with Python, Flask, HTML, CSS and JavaScript.

## Project structure

```text
priyamvadhaa_portfolio/
├── app.py
├── requirements.txt
├── Procfile
├── templates/
│   └── index.html
└── static/
    ├── css/
    │   └── style.css
    ├── js/
    │   └── script.js
    ├── images/       # Add your profile photo, project screenshots and videos here
    └── files/
        └── resume.pdf
```

## Run locally (Windows)

1. Install Python 3.10 or newer.
2. Open this folder in VS Code.
3. Open the terminal in this folder and run:

   ```powershell
   python -m venv .venv
   .venv\\Scripts\\activate
   pip install -r requirements.txt
   python app.py
   ```

4. Open **http://127.0.0.1:5000** in your browser.

The Flask development server listens on `0.0.0.0:5000`, so another device on the same Wi-Fi can use `http://YOUR_COMPUTER_LOCAL_IP:5000`. Find the IP with `ipconfig` on Windows (look for IPv4 Address). Allow Python through Windows Firewall if prompted.

## Production hosting

The included `Procfile` uses Gunicorn for platforms such as Render or Railway:

```bash
gunicorn app:app
```

Gunicorn is intended for Linux-based hosting and is not the recommended local Windows server. For a production-style local Windows run, install the requirements and use:

```powershell
waitress-serve --listen=0.0.0.0:5000 app:app
```

## Personalize

- Add a profile photo as `static/images/profile.jpg` and uncomment the image element in `templates/index.html`.
- Add project screenshots or demo videos to `static/images/` and update the relevant project cards.
- The resume PDF is included under `static/files/resume.pdf`.
- Update links, email, phone number and project details in `templates/index.html` as needed.
