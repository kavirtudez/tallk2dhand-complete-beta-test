import os
import subprocess
import sys
import time
import shutil

def start_flask_app():
    print("Starting Flask app...")
    flask_process = subprocess.Popen([sys.executable, "app.py"])
    return flask_process

def setup_angular_app():
    print("Setting up Angular app...")
    # Change directory to where your Angular app is located
    angular_app_dir = os.path.join(os.getcwd(), "ai-converse")  # Update this path as needed
    
    # Check if the directory exists
    if not os.path.exists(angular_app_dir):
        print(f"Error: Angular app directory not found at {angular_app_dir}")
        print("Please make sure the Angular project is in the correct location.")
        return False
    
    os.chdir(angular_app_dir)
    
    # Check if node_modules exists
    if not os.path.exists(os.path.join(angular_app_dir, "node_modules")):
        print("Installing Angular dependencies (this may take a few minutes)...")
        result = subprocess.run(["npm", "install"], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error installing dependencies: {result.stderr}")
            os.chdir(os.path.dirname(os.path.abspath(__file__)))  # Go back to original directory
            return False
        print("Dependencies installed successfully.")
    
    # Check if dist/ai-converse/browser exists
    if not os.path.exists(os.path.join(angular_app_dir, "dist", "ai-converse", "browser")):
        print("Building Angular app (this may take a few minutes)...")
        result = subprocess.run(["npm", "run", "build"], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error building Angular app: {result.stderr}")
            os.chdir(os.path.dirname(os.path.abspath(__file__)))  # Go back to original directory
            return False
        print("Angular app built successfully.")
    
    # Go back to original directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    return True

def start_angular_app():
    print("Starting Angular app with npx serve...")
    # Change directory to where your Angular app is located
    angular_app_dir = os.path.join(os.getcwd(), "ai-converse")  # Update this path as needed
    
    os.chdir(angular_app_dir)
    
    # Start the Angular app with npx serve
    angular_process = subprocess.Popen(["npx", "serve", "dist/ai-converse/browser", "-p", "3000"])
    
    # Go back to original directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    return angular_process

if __name__ == "__main__":
    print("=" * 80)
    print("Talk2DHand with AI Converse Startup Script")
    print("=" * 80)
    
    # Setup Angular app first
    if not setup_angular_app():
        print("Failed to set up Angular app. Exiting.")
        sys.exit(1)
    
    # Start both applications
    flask_app = start_flask_app()
    time.sleep(2)  # Give Flask app time to start
    angular_app = start_angular_app()
    
    try:
        print("\n" + "=" * 80)
        print("Both applications are running!")
        print("=" * 80)
        print("Flask app (Talk2DHand): http://127.0.0.1:5000")
        print("Angular app (AI Converse): http://localhost:3000")
        print("\nYou can access AI Converse through the Talk2DHand interface by clicking the 'AI Converse' button")
        print("\nPress Ctrl+C to stop both applications")
        
        # Keep the script running until interrupted
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping applications...")
        if flask_app:
            flask_app.terminate()
        if angular_app:
            angular_app.terminate()
        print("Applications stopped") 