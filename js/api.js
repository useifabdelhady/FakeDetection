/**
 * TruthLens — Shared API Module
 * Handles all communication with the ASP.NET Core backend.
 */

const API = (() => {
  // ─── Configuration ───────────────────────────────────────
  // Auto-detect: use local URLs in development, cloud URLs in production
  const isLocal = ["localhost", "127.0.0.1"].includes(window.location.hostname);

  // ⚠️ DEPLOYMENT: Replace these with your actual deployed URLs
  const PRODUCTION_BASE_URL = "https://truthlens-api.azurewebsites.net/api";
  const PRODUCTION_ML_URL   = "https://useifabdelhady-truthlens-ml.hf.space";

  const BASE_URL = isLocal ? "http://localhost:5159/api"  : PRODUCTION_BASE_URL;
  const ML_URL   = isLocal ? "http://localhost:8000"      : PRODUCTION_ML_URL;

  // ─── Token Management ────────────────────────────────────
  function getToken()   { return localStorage.getItem("jwtToken"); }
  function setToken(t)  { localStorage.setItem("jwtToken", t); }
  function clearToken() { localStorage.removeItem("jwtToken"); }

  function getUserInfo() {
    return {
      token:    getToken(),
      userId:   localStorage.getItem("userId"),
      username: localStorage.getItem("username"),
      email:    localStorage.getItem("email"),
      role:     localStorage.getItem("role"),
    };
  }

  function setUserInfo(data) {
    if (data.token)    localStorage.setItem("jwtToken", data.token);
    if (data.userId)   localStorage.setItem("userId", data.userId);
    if (data.username) localStorage.setItem("username", data.username);
    if (data.email)    localStorage.setItem("email", data.email);
    if (data.role)     localStorage.setItem("role", data.role);
  }

  function clearUserInfo() {
    ["jwtToken","userId","username","email","role","authType"].forEach(k => localStorage.removeItem(k));
  }

  function isLoggedIn() {
    return !!getToken();
  }

  // ─── HTTP Helper ─────────────────────────────────────────
  async function request(method, url, body = null, auth = true) {
    const headers = { "Content-Type": "application/json" };
    if (auth && getToken()) {
      headers["Authorization"] = "Bearer " + getToken();
    }

    const opts = { method, headers };
    if (body) opts.body = JSON.stringify(body);

    try {
      const res = await fetch(url, opts);
      const text = await res.text();
      let data;
      try { data = JSON.parse(text); } catch { data = text; }

      if (!res.ok) {
        // ASP.NET validation errors come as { errors: { FieldName: ["message"] } }
        if (data?.errors) {
          const messages = Object.values(data.errors).flat().join(". ");
          throw new Error(messages || data.title || `Error ${res.status}`);
        }
        const msg = data?.message || data?.title || (typeof data === 'string' ? data : `Error ${res.status}`);
        throw new Error(msg);
      }
      return data;
    } catch (err) {
      if (err.message.includes("Failed to fetch") || err.message.includes("NetworkError")) {
        throw new Error("Cannot connect to the server. Make sure the backend is running.");
      }
      throw err;
    }
  }

  // ─── Auth Endpoints ──────────────────────────────────────
  async function login(email, password) {
    const data = await request("POST", `${BASE_URL}/Auth/login`, { Email: email, Password: password }, false);
    setUserInfo({
      token: data.token || data.Token,
      userId: (data.userId || data.UserId)?.toString(),
      username: data.username || data.Username,
      email: data.email || data.Email,
      role: data.role || data.Role,
    });
    localStorage.setItem("authType", "signin");
    return data;
  }

  async function register(username, email, password, confirmPassword) {
    const data = await request("POST", `${BASE_URL}/Auth/register`, {
      Username: username, Email: email, Password: password, ConfirmPassword: confirmPassword
    }, false);
    setUserInfo({
      token: data.token || data.Token,
      userId: (data.userId || data.UserId)?.toString(),
      username: data.username || data.Username,
      email: data.email || data.Email,
      role: data.role || data.Role || "User",
    });
    localStorage.setItem("authType", "signup");
    return data;
  }

  function logout() {
    clearUserInfo();
    window.location.href = "TruthLens.html";
  }

  // ─── Analysis Endpoints (via ASP.NET Backend) ────────
  // The C# backend will call the Python ML service and save the results

  async function analyzeNews(content) {
    const data = await request("POST", `${BASE_URL}/NewsAnalysis/analyze`, {
      Content: content
    }, true);
    return data;
  }

  async function analyzeWebsite(url, deepScan = false) {
    const data = await request("POST", `${BASE_URL}/WebsiteAnalysis/analyze`, {
      Url: url
    }, true);
    return data;
  }

  // ─── Backend API Endpoints (for user data, history, etc.) ─
  async function getProfile() {
    return await request("GET", `${BASE_URL}/Auth/profile`);
  }

  async function changePassword(currentPassword, newPassword) {
    return await request("POST", `${BASE_URL}/Auth/change-password`, {
      currentPassword, newPassword
    });
  }

  async function updateProfile(username) {
    return await request("PUT", `${BASE_URL}/Auth/profile`, {
      username
    });
  }

  async function getMyAnalysisRequests() {
    return await request("GET", `${BASE_URL}/Analysis/my-requests`);
  }

  async function getMyDashboard() {
    return await request("GET", `${BASE_URL}/Statistics/my-dashboard`);
  }

  async function submitFeedback(analysisType, analysisId, rating, isAccurate, comments) {
    return await request("POST", `${BASE_URL}/Feedback`, {
      AnalysisType: analysisType,
      AnalysisId: analysisId,
      Rating: rating,
      IsAccurate: isAccurate,
      Comments: comments
    }, true);
  }

  // ─── Snackbar UI Utility ──────────────────────────────────
  function showSnackbar(message, type = 'info') {
    // Determine colors
    let bgColor = '#0A2540'; // info
    if (type === 'error') bgColor = '#E74C3C';
    if (type === 'success') bgColor = '#2ECC71';
    if (type === 'warning') bgColor = '#F1C40F';

    // Create container if it doesn't exist
    let container = document.getElementById('snackbar-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'snackbar-container';
        container.style.cssText = `
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            display: flex;
            flex-direction: column;
            gap: 10px;
            z-index: 9999;
            pointer-events: none;
        `;
        document.body.appendChild(container);
    }

    // Create snackbar el
    const snackbar = document.createElement('div');
    snackbar.textContent = message;
    snackbar.style.cssText = `
        background-color: ${bgColor};
        color: white;
        padding: 12px 24px;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        font-family: inherit;
        font-size: 14px;
        font-weight: 500;
        opacity: 0;
        transform: translateY(20px);
        transition: all 0.3s ease-out;
    `;
    container.appendChild(snackbar);

    // Animate in
    requestAnimationFrame(() => {
        snackbar.style.opacity = '1';
        snackbar.style.transform = 'translateY(0)';
    });

    // Animate out and remove
    setTimeout(() => {
        snackbar.style.opacity = '0';
        snackbar.style.transform = 'translateY(20px)';
        snackbar.addEventListener('transitionend', () => snackbar.remove());
    }, 3000);
  }

  // Expose globally
  window.showSnackbar = showSnackbar;

  // ─── Public API ──────────────────────────────────────────
  return {
    BASE_URL, ML_URL,
    getToken, isLoggedIn, getUserInfo, clearUserInfo,
    login, register, logout,
    analyzeNews, analyzeWebsite,
    getProfile, changePassword, updateProfile, getMyAnalysisRequests, getMyDashboard, submitFeedback,
  };
})();
