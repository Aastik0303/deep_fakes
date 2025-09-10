from flask import Flask, render_template_string, request, jsonify, redirect, url_for, session
import uuid  # For generating unique user IDs

# --- FLASK APPLICATION SETUP ---
app = Flask(__name__)
# A secret key is required to secure the session object
app.config['SECRET_KEY'] = 'a_very_secret_key_that_you_should_change'

# A dictionary to simulate a user database in memory
# In a real application, this would be a database query
users = {
    'user': {'id': 'user_id_1', 'name': 'Demonstration User', 'password': 'password'}
}

# --- HTML TEMPLATE STRING WITH EMBEDDED CSS AND JS ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ page_title }}</title>
    <style>
        :root {
            --primary-color: #6366f1;
            --primary-hover: #5b21b6;
            --success-color: #10b981;
            --success-hover: #059669;
            --error-color: #ef4444;
            --warning-color: #f59e0b;
            --text-primary: #1f2937;
            --text-secondary: #6b7280;
            --bg-primary: #f8fafc;
            --bg-secondary: #ffffff;
            --border-color: #e5e7eb;
            --border-focus: #6366f1;
            --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
            --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
            --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
            --shadow-xl: 0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1);
            --radius: 0.5rem;
            --radius-lg: 0.75rem;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 1rem;
            line-height: 1.6;
            color: var(--text-primary);
        }

        .container {
            background: var(--bg-secondary);
            padding: 2.5rem;
            border-radius: var(--radius-lg);
            box-shadow: var(--shadow-xl);
            width: 100%;
            max-width: 28rem;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .logo {
            text-align: center;
            margin-bottom: 2rem;
        }

        .logo-icon {
            width: 3rem;
            height: 3rem;
            background: linear-gradient(135deg, var(--primary-color), var(--success-color));
            border-radius: 50%;
            margin: 0 auto 1rem;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 1.5rem;
            font-weight: bold;
        }

        h1 {
            color: var(--text-primary);
            font-size: 1.875rem;
            font-weight: 700;
            text-align: center;
            margin-bottom: 0.5rem;
        }

        .subtitle {
            color: var(--text-secondary);
            text-align: center;
            margin-bottom: 2rem;
            font-size: 0.875rem;
        }

        /* Logged in state */
        .welcome-section {
            text-align: center;
        }

        .user-avatar {
            width: 4rem;
            height: 4rem;
            background: linear-gradient(135deg, var(--primary-color), var(--success-color));
            border-radius: 50%;
            margin: 0 auto 1rem;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 1.5rem;
            font-weight: bold;
        }

        .welcome-message {
            margin-bottom: 2rem;
        }

        .protected-content {
            background: #f8fafc;
            padding: 1.5rem;
            border-radius: var(--radius);
            margin-top: 1.5rem;
            border-left: 4px solid var(--success-color);
        }

        /* Button styles */
        .btn-group {
            display: flex;
            gap: 0.75rem;
            justify-content: center;
            margin-bottom: 1rem;
        }

        .btn {
            padding: 0.75rem 1.5rem;
            border: none;
            border-radius: var(--radius);
            font-size: 0.875rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease-in-out;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            min-width: 7rem;
        }

        .btn-primary {
            background: var(--primary-color);
            color: white;
            box-shadow: var(--shadow-sm);
        }

        .btn-primary:hover {
            background: var(--primary-hover);
            box-shadow: var(--shadow-md);
            transform: translateY(-1px);
        }

        .btn-success {
            background: var(--success-color);
            color: white;
            box-shadow: var(--shadow-sm);
        }

        .btn-success:hover {
            background: var(--success-hover);
            box-shadow: var(--shadow-md);
            transform: translateY(-1px);
        }

        .btn-outline {
            background: transparent;
            color: var(--text-secondary);
            border: 1px solid var(--border-color);
        }

        .btn-outline:hover {
            background: var(--bg-primary);
            color: var(--text-primary);
            border-color: var(--border-focus);
        }

        /* Modal styles */
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.6);
            backdrop-filter: blur(4px);
            align-items: center;
            justify-content: center;
            padding: 1rem;
        }

        .modal.show {
            display: flex;
        }

        .modal-content {
            background: var(--bg-secondary);
            padding: 2rem;
            border-radius: var(--radius-lg);
            box-shadow: var(--shadow-xl);
            width: 100%;
            max-width: 24rem;
            animation: modalSlideIn 0.3s ease-out;
            position: relative;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        @keyframes modalSlideIn {
            from {
                opacity: 0;
                transform: translateY(-2rem) scale(0.95);
            }
            to {
                opacity: 1;
                transform: translateY(0) scale(1);
            }
        }

        .modal-header {
            display: flex;
            justify-content: between;
            align-items: center;
            margin-bottom: 1.5rem;
        }

        .modal-title {
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--text-primary);
        }

        .close-btn {
            position: absolute;
            top: 1rem;
            right: 1rem;
            background: none;
            border: none;
            font-size: 1.5rem;
            cursor: pointer;
            color: var(--text-secondary);
            padding: 0.25rem;
            border-radius: var(--radius);
            transition: all 0.2s ease;
            width: 2rem;
            height: 2rem;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .close-btn:hover {
            background: var(--bg-primary);
            color: var(--text-primary);
        }

        /* Form styles */
        .form {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }

        .form-label {
            font-size: 0.875rem;
            font-weight: 500;
            color: var(--text-primary);
        }

        .form-input {
            padding: 0.75rem;
            border: 1px solid var(--border-color);
            border-radius: var(--radius);
            font-size: 1rem;
            transition: all 0.2s ease;
            background: var(--bg-secondary);
        }

        .form-input:focus {
            outline: none;
            border-color: var(--border-focus);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
        }

        .form-input::placeholder {
            color: var(--text-secondary);
        }

        /* Message styles */
        .message {
            padding: 0.75rem;
            border-radius: var(--radius);
            font-size: 0.875rem;
            margin-top: 1rem;
            display: none;
        }

        .message.show {
            display: block;
        }

        .message.error {
            background: #fef2f2;
            color: #991b1b;
            border: 1px solid #fecaca;
        }

        .message.success {
            background: #f0fdf4;
            color: #166534;
            border: 1px solid #bbf7d0;
        }

        /* Loading state */
        .btn.loading {
            opacity: 0.7;
            cursor: not-allowed;
            position: relative;
        }

        .btn.loading::after {
            content: '';
            position: absolute;
            width: 1rem;
            height: 1rem;
            border: 2px solid transparent;
            border-top: 2px solid currentColor;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-left: 0.5rem;
        }

        @keyframes spin {
            to {
                transform: rotate(360deg);
            }
        }

        /* Responsive design */
        @media (max-width: 640px) {
            .container {
                padding: 1.5rem;
                margin: 0.5rem;
            }

            .btn-group {
                flex-direction: column;
            }

            .modal-content {
                padding: 1.5rem;
                margin: 1rem;
            }
        }

        /* Focus indicators for accessibility */
        .btn:focus-visible,
        .form-input:focus-visible {
            outline: 2px solid var(--border-focus);
            outline-offset: 2px;
        }

        /* Dark mode support */
        @media (prefers-color-scheme: dark) {
            :root {
                --text-primary: #f9fafb;
                --text-secondary: #9ca3af;
                --bg-primary: #111827;
                --bg-secondary: #1f2937;
                --border-color: #374151;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        {% if 'logged_in' in session and session['logged_in'] %}
            <div class="welcome-section">
                <div class="logo">
                    <div class="user-avatar">{{ session['user_name'][0].upper() }}</div>
                </div>
                
                <div class="welcome-message">
                    <h1>Welcome back, {{ session['user_name'] }}!</h1>
                    <p class="subtitle">You're successfully logged in to your account.</p>
                </div>
                
                <button class="btn btn-outline" onclick="logout()">
                    <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1"/>
                    </svg>
                    Sign Out
                </button>
                
                <div class="protected-content">
                    <h3 style="margin-bottom: 0.5rem; color: var(--text-primary);">🎉 Access Granted</h3>
                    <p style="color: var(--text-secondary); font-size: 0.875rem;">
                        This is your protected dashboard. Your secure content and features would be displayed here.
                    </p>
                </div>
            </div>
        {% else %}
            <div class="logo">
                <div class="logo-icon">🔐</div>
                <h1>Welcome</h1>
                <p class="subtitle">Sign in to your account or create a new one to get started.</p>
            </div>
            
            <div class="btn-group">
                <button id="show-login-btn" class="btn btn-primary">
                    <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 16l-4-4m0 0l4-4m0 4h14m-5 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h7a3 3 0 013 3v1"/>
                    </svg>
                    Sign In
                </button>
                <button id="show-register-btn" class="btn btn-success">
                    <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M18 9v3m0 0v3m0-3h3m-3 0h-3m-2-5a4 4 0 11-8 0 4 4 0 018 0zM3 20a6 6 0 0112 0v1H3v-1z"/>
                    </svg>
                    Register
                </button>
            </div>
        {% endif %}
    </div>

    <!-- Login Modal -->
    <div id="login-modal" class="modal" role="dialog" aria-labelledby="login-title" aria-modal="true">
        <div class="modal-content">
            <button class="close-btn login-close-btn" aria-label="Close login modal">&times;</button>
            
            <div class="modal-header">
                <h2 id="login-title" class="modal-title">Sign In</h2>
            </div>
            
            <form id="login-form" class="form" novalidate>
                <div class="form-group">
                    <label for="login-username" class="form-label">Username</label>
                    <input type="text" 
                           id="login-username" 
                           name="username" 
                           class="form-input" 
                           placeholder="Enter your username" 
                           autocomplete="username"
                           required 
                           aria-describedby="login-error-message">
                </div>
                
                <div class="form-group">
                    <label for="login-password" class="form-label">Password</label>
                    <input type="password" 
                           id="login-password" 
                           name="password" 
                           class="form-input" 
                           placeholder="Enter your password"
                           autocomplete="current-password" 
                           required
                           aria-describedby="login-error-message">
                </div>
                
                <button type="submit" class="btn btn-primary" style="margin-top: 0.5rem;">
                    Sign In
                </button>
            </form>
            
            <div class="message error" id="login-error-message" role="alert" aria-live="polite"></div>
        </div>
    </div>

    <!-- Register Modal -->
    <div id="register-modal" class="modal" role="dialog" aria-labelledby="register-title" aria-modal="true">
        <div class="modal-content">
            <button class="close-btn register-close-btn" aria-label="Close registration modal">&times;</button>
            
            <div class="modal-header">
                <h2 id="register-title" class="modal-title">Create Account</h2>
            </div>
            
            <form id="register-form" class="form" novalidate>
                <div class="form-group">
                    <label for="register-username" class="form-label">Choose Username</label>
                    <input type="text" 
                           id="register-username" 
                           name="username" 
                           class="form-input" 
                           placeholder="Enter a unique username"
                           autocomplete="username"
                           required 
                           aria-describedby="register-error-message">
                </div>
                
                <div class="form-group">
                    <label for="register-password" class="form-label">Create Password</label>
                    <input type="password" 
                           id="register-password" 
                           name="password" 
                           class="form-input" 
                           placeholder="Create a secure password"
                           autocomplete="new-password"
                           required
                           aria-describedby="register-error-message">
                </div>
                
                <button type="submit" class="btn btn-success" style="margin-top: 0.5rem;">
                    Create Account
                </button>
            </form>
            
            <div class="message error" id="register-error-message" role="alert" aria-live="polite"></div>
        </div>
    </div>

    <script>
        class AuthManager {
            constructor() {
                this.initializeElements();
                this.attachEventListeners();
            }

            initializeElements() {
                // Modals
                this.loginModal = document.getElementById('login-modal');
                this.registerModal = document.getElementById('register-modal');

                // Buttons
                this.showLoginBtn = document.getElementById('show-login-btn');
                this.showRegisterBtn = document.getElementById('show-register-btn');
                this.loginCloseBtn = document.querySelector('.login-close-btn');
                this.registerCloseBtn = document.querySelector('.register-close-btn');

                // Forms
                this.loginForm = document.getElementById('login-form');
                this.registerForm = document.getElementById('register-form');

                // Error messages
                this.loginErrorMsg = document.getElementById('login-error-message');
                this.registerErrorMsg = document.getElementById('register-error-message');
            }

            attachEventListeners() {
                // Modal show/hide
                if (this.showLoginBtn) {
                    this.showLoginBtn.addEventListener('click', () => this.showModal('login'));
                }
                
                if (this.showRegisterBtn) {
                    this.showRegisterBtn.addEventListener('click', () => this.showModal('register'));
                }

                if (this.loginCloseBtn) {
                    this.loginCloseBtn.addEventListener('click', () => this.hideModal('login'));
                }

                if (this.registerCloseBtn) {
                    this.registerCloseBtn.addEventListener('click', () => this.hideModal('register'));
                }

                // Close on backdrop click
                window.addEventListener('click', (event) => {
                    if (event.target === this.loginModal) {
                        this.hideModal('login');
                    }
                    if (event.target === this.registerModal) {
                        this.hideModal('register');
                    }
                });

                // Close on escape key
                document.addEventListener('keydown', (event) => {
                    if (event.key === 'Escape') {
                        this.hideModal('login');
                        this.hideModal('register');
                    }
                });

                // Form submissions
                if (this.loginForm) {
                    this.loginForm.addEventListener('submit', (e) => this.handleLogin(e));
                }

                if (this.registerForm) {
                    this.registerForm.addEventListener('submit', (e) => this.handleRegister(e));
                }
            }

            showModal(type) {
                const modal = type === 'login' ? this.loginModal : this.registerModal;
                if (modal) {
                    modal.classList.add('show');
                    // Focus first input for accessibility
                    const firstInput = modal.querySelector('input[type="text"]');
                    if (firstInput) {
                        setTimeout(() => firstInput.focus(), 100);
                    }
                }
            }

            hideModal(type) {
                const modal = type === 'login' ? this.loginModal : this.registerModal;
                const errorMsg = type === 'login' ? this.loginErrorMsg : this.registerErrorMsg;
                
                if (modal) {
                    modal.classList.remove('show');
                }
                if (errorMsg) {
                    this.hideMessage(errorMsg);
                }
            }

            showMessage(element, message, isError = true) {
                if (element) {
                    element.textContent = message;
                    element.className = `message ${isError ? 'error' : 'success'} show`;
                }
            }

            hideMessage(element) {
                if (element) {
                    element.classList.remove('show');
                    element.textContent = '';
                }
            }

            setButtonLoading(button, isLoading) {
                if (button) {
                    if (isLoading) {
                        button.classList.add('loading');
                        button.disabled = true;
                    } else {
                        button.classList.remove('loading');
                        button.disabled = false;
                    }
                }
            }

            async handleLogin(e) {
                e.preventDefault();
                
                const submitBtn = this.loginForm.querySelector('button[type="submit"]');
                this.setButtonLoading(submitBtn, true);
                this.hideMessage(this.loginErrorMsg);

                try {
                    const formData = new FormData(this.loginForm);
                    const data = Object.fromEntries(formData.entries());

                    const response = await fetch('{{ url_for('login') }}', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(data),
                    });

                    const result = await response.json();

                    if (result.success) {
                        this.showMessage(this.loginErrorMsg, 'Login successful! Redirecting...', false);
                        setTimeout(() => window.location.reload(), 1000);
                    } else {
                        this.showMessage(this.loginErrorMsg, result.message || 'Login failed. Please try again.');
                    }
                } catch (error) {
                    console.error('Login error:', error);
                    this.showMessage(this.loginErrorMsg, 'Network error. Please check your connection and try again.');
                } finally {
                    this.setButtonLoading(submitBtn, false);
                }
            }

            async handleRegister(e) {
                e.preventDefault();
                
                const submitBtn = this.registerForm.querySelector('button[type="submit"]');
                this.setButtonLoading(submitBtn, true);
                this.hideMessage(this.registerErrorMsg);

                try {
                    const formData = new FormData(this.registerForm);
                    const data = Object.fromEntries(formData.entries());

                    // Basic client-side validation
                    if (data.username.length < 3) {
                        this.showMessage(this.registerErrorMsg, 'Username must be at least 3 characters long.');
                        return;
                    }

                    if (data.password.length < 6) {
                        this.showMessage(this.registerErrorMsg, 'Password must be at least 6 characters long.');
                        return;
                    }

                    const response = await fetch('{{ url_for('register') }}', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(data),
                    });

                    const result = await response.json();

                    if (result.success) {
                        this.showMessage(this.registerErrorMsg, 'Account created successfully! Logging you in...', false);
                        setTimeout(() => window.location.reload(), 1000);
                    } else {
                        this.showMessage(this.registerErrorMsg, result.message || 'Registration failed. Please try again.');
                    }
                } catch (error) {
                    console.error('Registration error:', error);
                    this.showMessage(this.registerErrorMsg, 'Network error. Please check your connection and try again.');
                } finally {
                    this.setButtonLoading(submitBtn, false);
                }
            }
        }

        // Global logout function
        function logout() {
            if (confirm('Are you sure you want to sign out?')) {
                window.location.href = '{{ url_for('logout') }}';
            }
        }

        // Initialize when DOM is ready
        document.addEventListener('DOMContentLoaded', () => {
            new AuthManager();
        });
    </script>
</body>
</html>"""


# --- FLASK ROUTES ---

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'logged_in' in session and session['logged_in']:
        return redirect(url_for('index'))

    if request.method == 'POST':
        if not request.is_json:
            return jsonify({"success": False, "message": "Request must be JSON"}), 400

        username = request.json.get('username')
        password = request.json.get('password')

        user = users.get(username)
        if user and user['password'] == password:
            session['logged_in'] = True
            session['user_id'] = user['id']
            session['user_name'] = user['name']
            return jsonify({"success": True, "message": "Login successful"})
        else:
            return jsonify({"success": False, "message": "Invalid username or password"})

    return render_template_string(HTML_TEMPLATE, page_title="Login & Register", session=session)


@app.route('/register', methods=['POST'])
def register():
    if not request.is_json:
        return jsonify({"success": False, "message": "Request must be JSON"}), 400

    username = request.json.get('username')
    password = request.json.get('password')

    if username in users:
        return jsonify({"success": False, "message": "Username already exists."})

    # Simple validation
    if not username or not password or len(password) < 4:
        return jsonify({"success": False,
                        "message": "Username and password are required. Password must be at least 4 characters."})

    # Add the new user to the in-memory dictionary
    new_user_id = str(uuid.uuid4())
    users[username] = {
        'id': new_user_id,
        'name': username,
        'password': password
    }

    # Log the new user in automatically
    session['logged_in'] = True
    session['user_id'] = new_user_id
    session['user_name'] = username

    return jsonify({"success": True, "message": "Registration successful"})


@app.route('/')
def index():
    if 'logged_in' not in session or not session['logged_in']:
        return redirect(url_for('login'))

    return render_template_string(HTML_TEMPLATE, page_title="Dashboard", session=session)


@app.route('/logout')
def logout():
    session.clear()  # Clear all session variables
    return redirect(url_for('login'))


if __name__ == '__main__':
    # You must have Flask installed: pip install Flask
    app.run(debug=True)
