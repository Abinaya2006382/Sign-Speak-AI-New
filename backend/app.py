import os
import sys
from flask import Flask, send_from_directory
from flask_cors import CORS

# Add project root directory to path to resolve imports correctly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from backend.database import db_manager
from backend.routes.api import api_bp

def create_app():
    # Resolve absolute path to the frontend folder
    frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))
    
    # Initialize Flask app serving assets from the frontend directory
    app = Flask(__name__, static_folder=frontend_dir, static_url_path='')
    
    # Enable CORS for standard web app usage
    CORS(app)
    
    # Initialize Database
    db_manager.init_db()
    
    # Register REST API Blueprints
    app.register_blueprint(api_bp, url_prefix='/api')
    
    # Serve index.html as the primary landing page
    @app.route('/')
    def index():
        return send_from_directory(app.static_folder, 'index.html')
        
    # Serve other HTML sections/views if requested directly
    @app.route('/<path:path>')
    def serve_static(path):
        if os.path.exists(os.path.join(app.static_folder, path)):
            return send_from_directory(app.static_folder, path)
        return send_from_directory(app.static_folder, 'index.html')
        
    return app

if __name__ == '__main__':
    app = create_app()
    port = int(os.environ.get('PORT', 5000))
    print(f"Sign Speak AI backend server booting on http://localhost:{port}")
    # Run server locally, multi-threaded to prevent blocking on predictions/speech
    app.run(host='0.0.0.0', port=port, debug=True, threaded=True)
