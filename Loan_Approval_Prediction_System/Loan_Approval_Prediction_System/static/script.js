/* ==========================================================================
   CreditPulse AI - Application Logic, Backend APIs & Authentication
   ========================================================================== */

(function () {
  "use strict";

  // State
  let currentUser = null;
  const data = window.LOAN_MODEL_DATA;

  // DOM Elements
  const form = document.getElementById("applicant-form");
  const fieldsHost = document.getElementById("form-fields");
  const resultsHost = document.getElementById("result-list");
  const consensusBanner = document.getElementById("consensus-banner");
  const consensusIcon = document.getElementById("consensus-icon");
  const consensusStatus = document.getElementById("consensus-status");
  const consensusDetail = document.getElementById("consensus-detail");
  const driverCard = document.getElementById("driver-card");
  const driverList = document.getElementById("driver-list");
  const metricsHost = document.getElementById("metrics-body");
  const presetContainer = document.getElementById("preset-buttons");
  const fillDefaultsBtn = document.getElementById("fill-defaults-btn");
  const printReportBtn = document.getElementById("print-report-btn");
  const postPredActions = document.getElementById("post-pred-actions");
  const scrollToHistoryBtn = document.getElementById("scroll-to-history-btn");
  const newAppBtn = document.getElementById("new-app-btn");
  const authSaveIndicator = document.getElementById("auth-save-indicator");

  // Auth & Navigation Elements
  const authGuestLinks = document.getElementById("auth-guest-links");
  const authUserMenu = document.getElementById("auth-user-menu");
  const userNameDisplay = document.getElementById("user-name-display");
  const userAvatar = document.getElementById("user-avatar");
  const adminBadge = document.getElementById("admin-badge");
  const logoutBtn = document.getElementById("logout-btn");
  const openLoginBtn = document.getElementById("open-login-btn");
  const openRegisterBtn = document.getElementById("open-register-btn");
  const loginModal = document.getElementById("login-modal");
  const registerModal = document.getElementById("register-modal");
  const detailModal = document.getElementById("detail-modal");
  const detailModalBody = document.getElementById("detail-modal-body");
  const switchToRegister = document.getElementById("switch-to-register");
  const switchToLogin = document.getElementById("switch-to-login");
  const loginForm = document.getElementById("login-form");
  const registerForm = document.getElementById("register-form");
  const loginAlert = document.getElementById("login-alert");
  const registerAlert = document.getElementById("register-alert");
  const toastContainer = document.getElementById("toast-container");

  // Dashboard & Admin Elements
  const userDashboardSection = document.getElementById("dashboard");
  const adminSection = document.getElementById("admin");
  const authRequiredNavs = document.querySelectorAll(".auth-required-nav");
  const adminRequiredNavs = document.querySelectorAll(".admin-required-nav");
  const userTotalApps = document.getElementById("user-total-apps");
  const userApprovedApps = document.getElementById("user-approved-apps");
  const userRejectedApps = document.getElementById("user-rejected-apps");
  const userDbEngine = document.getElementById("user-db-engine");
  const userAppsTbody = document.getElementById("user-apps-tbody");

  // Admin Elements
  const adminUserCount = document.getElementById("admin-user-count");
  const adminAppCount = document.getElementById("admin-app-count");
  const adminDbStatus = document.getElementById("admin-db-status");
  const adminDbName = document.getElementById("admin-db-name");
  const adminPredCountsTbody = document.getElementById("admin-pred-counts-tbody");
  const adminAppsTbody = document.getElementById("admin-apps-tbody");
  const adminSearchInput = document.getElementById("admin-search-input");
  const adminSearchBtn = document.getElementById("admin-search-btn");

  // Collateral elements
  const collateralRatioVal = document.getElementById("collateral-ratio-val");
  const collateralBar = document.getElementById("collateral-bar");
  const totalAssetsVal = document.getElementById("total-assets-val");
  const loanAmountDisp = document.getElementById("loan-amount-disp");

  const modelMeta = {
    random_forest: {
      name: "Random Forest Classifier",
      badge: "Champion · 98.1% Acc",
      isChampion: true,
      note: "Ensemble of 200 de-correlated trees with bootstrap aggregation"
    },
    decision_tree: {
      name: "Decision Tree (CART)",
      badge: "Rule-Based · 97.3% Acc",
      isChampion: false,
      note: "Recursive binary split with Gini impurity optimization"
    },
    logistic_regression: {
      name: "Logistic Regression",
      badge: "Linear Sigmoid · 92.3% Acc",
      isChampion: false,
      note: "L2-regularized log-loss with standardized numeric scaling"
    }
  };

  const featureLabels = {
    no_of_dependents: "Dependents",
    education: "Education Level",
    self_employed: "Self-Employed",
    income_annum: "Annual Income (₹)",
    loan_amount: "Requested Loan Amount (₹)",
    loan_term: "Loan Term (Years)",
    cibil_score: "CIBIL Credit Score (300 - 900)",
    residential_assets_value: "Residential Assets (₹)",
    commercial_assets_value: "Commercial Assets (₹)",
    luxury_assets_value: "Luxury Assets (₹)",
    bank_asset_value: "Bank Balance & Liquid Assets (₹)",
  };

  const formGroups = [
    {
      title: "Applicant Background",
      features: ["no_of_dependents", "education", "self_employed"]
    },
    {
      title: "Loan Financials",
      features: ["income_annum", "loan_amount", "loan_term"]
    },
    {
      title: "Credit History & Assets",
      features: [
        "cibil_score",
        "residential_assets_value",
        "commercial_assets_value",
        "luxury_assets_value",
        "bank_asset_value"
      ]
    }
  ];

  const presets = {
    prime: {
      no_of_dependents: 1,
      education: "Graduate",
      self_employed: "No",
      income_annum: 9500000,
      loan_amount: 25000000,
      loan_term: 10,
      cibil_score: 820,
      residential_assets_value: 18000000,
      commercial_assets_value: 12000000,
      luxury_assets_value: 24000000,
      bank_asset_value: 8500000
    },
    entrepreneur: {
      no_of_dependents: 2,
      education: "Graduate",
      self_employed: "Yes",
      income_annum: 8000000,
      loan_amount: 22000000,
      loan_term: 8,
      cibil_score: 740,
      residential_assets_value: 12000000,
      commercial_assets_value: 16000000,
      luxury_assets_value: 18000000,
      bank_asset_value: 6200000
    },
    borderline: {
      no_of_dependents: 3,
      education: "Not Graduate",
      self_employed: "No",
      income_annum: 4200000,
      loan_amount: 18000000,
      loan_term: 18,
      cibil_score: 590,
      residential_assets_value: 4000000,
      commercial_assets_value: 2000000,
      luxury_assets_value: 7000000,
      bank_asset_value: 2500000
    },
    highrisk: {
      no_of_dependents: 4,
      education: "Not Graduate",
      self_employed: "Yes",
      income_annum: 2200000,
      loan_amount: 16000000,
      loan_term: 20,
      cibil_score: 380,
      residential_assets_value: 2000000,
      commercial_assets_value: 1000000,
      luxury_assets_value: 3000000,
      bank_asset_value: 1100000
    }
  };

  const metadata = data ? data.metadata : {
    numeric_features: [
      "no_of_dependents", "income_annum", "loan_amount", "loan_term",
      "cibil_score", "residential_assets_value", "commercial_assets_value",
      "luxury_assets_value", "bank_asset_value"
    ],
    categorical_features: ["education", "self_employed"],
    categorical_values: {
      education: ["Graduate", "Not Graduate"],
      self_employed: ["No", "Yes"]
    },
    numeric_defaults: {
      no_of_dependents: 3, income_annum: 5100000, loan_amount: 14500000,
      loan_term: 10, cibil_score: 600, residential_assets_value: 5600000,
      commercial_assets_value: 3700000, luxury_assets_value: 14600000,
      bank_asset_value: 4600000
    },
    positive_class: "Approved",
    features: [
      "no_of_dependents", "education", "self_employed", "income_annum",
      "loan_amount", "loan_term", "cibil_score", "residential_assets_value",
      "commercial_assets_value", "luxury_assets_value", "bank_asset_value"
    ]
  };

  const numericSet = new Set(metadata.numeric_features);

  // ---------------------------------------------------------------------------
  // UI Helpers (Currency formatting, Toast, Modals)
  // ---------------------------------------------------------------------------

  function formatCurrency(val) {
    const num = Number(val);
    if (!Number.isFinite(num)) return "₹ 0";
    if (num >= 10000000) {
      return `₹ ${(num / 10000000).toFixed(2)} Cr`;
    } else if (num >= 100000) {
      return `₹ ${(num / 100000).toFixed(2)} Lakh`;
    }
    return `₹ ${num.toLocaleString("en-IN")}`;
  }

  function showToast(message, type = "success") {
    if (!toastContainer) return;
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    toast.textContent = message;
    toastContainer.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = "0";
      toast.style.transform = "translateY(10px)";
      toast.style.transition = "all 0.3s ease";
      setTimeout(() => toast.remove(), 300);
    }, 4000);
  }

  function openModal(modalEl) {
    if (modalEl) modalEl.style.display = "flex";
  }

  function closeModal(modalEl) {
    if (modalEl) modalEl.style.display = "none";
  }

  document.querySelectorAll(".modal-close-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const modalId = btn.dataset.close;
      closeModal(document.getElementById(modalId));
    });
  });

  document.querySelectorAll(".modal-backdrop").forEach((backdrop) => {
    backdrop.addEventListener("click", (e) => {
      if (e.target === backdrop) closeModal(backdrop);
    });
  });

  if (switchToRegister) {
    switchToRegister.addEventListener("click", () => {
      closeModal(loginModal);
      openModal(registerModal);
    });
  }

  if (switchToLogin) {
    switchToLogin.addEventListener("click", () => {
      closeModal(registerModal);
      openModal(loginModal);
    });
  }

  if (openLoginBtn) openLoginBtn.addEventListener("click", () => openModal(loginModal));
  if (openRegisterBtn) openRegisterBtn.addEventListener("click", () => openModal(registerModal));

  // ---------------------------------------------------------------------------
  // Authentication Management
  // ---------------------------------------------------------------------------

  async function checkAuthStatus() {
    try {
      const res = await fetch("/api/me");
      const auth = await res.json();
      if (auth.authenticated && auth.user) {
        currentUser = auth.user;
        renderAuthUI(true, auth.user, auth.database_engine);
        loadUserDashboard();
        if (auth.user.is_admin) {
          loadAdminPortal();
        }
      } else {
        currentUser = null;
        renderAuthUI(false, null, auth.database_engine);
      }
    } catch {
      currentUser = null;
      renderAuthUI(false, null, "local");
    }
  }

  function renderAuthUI(isLoggedIn, user, dbEngine) {
    if (isLoggedIn && user) {
      authGuestLinks.style.display = "none";
      authUserMenu.style.display = "flex";
      userNameDisplay.textContent = user.full_name;
      userAvatar.textContent = user.full_name.charAt(0).toUpperCase();

      if (user.is_admin) {
        adminBadge.style.display = "inline-block";
        adminRequiredNavs.forEach((n) => (n.style.display = "inline-block"));
      } else {
        adminBadge.style.display = "none";
        adminRequiredNavs.forEach((n) => (n.style.display = "none"));
      }

      authRequiredNavs.forEach((n) => (n.style.display = "inline-block"));
      if (userDashboardSection) userDashboardSection.style.display = "block";
      if (authSaveIndicator) {
        authSaveIndicator.className = "auth-save-hint saved";
        authSaveIndicator.textContent = "✓ Connected: Applications are saved to your account";
      }
    } else {
      authGuestLinks.style.display = "flex";
      authUserMenu.style.display = "none";
      authRequiredNavs.forEach((n) => (n.style.display = "none"));
      adminRequiredNavs.forEach((n) => (n.style.display = "none"));
      if (userDashboardSection) userDashboardSection.style.display = "none";
      if (adminSection) adminSection.style.display = "none";
      if (authSaveIndicator) {
        authSaveIndicator.className = "auth-save-hint";
        authSaveIndicator.textContent = "🔒 Log in to save to history";
      }
    }

    if (userDbEngine) userDbEngine.textContent = (dbEngine || "MySQL").toUpperCase();
  }

  // Register Form Submission
  if (registerForm) {
    registerForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      registerAlert.style.display = "none";
      const payload = Object.fromEntries(new FormData(registerForm).entries());

      try {
        const res = await fetch("/api/register", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        const resp = await res.json();
        if (res.ok) {
          closeModal(registerModal);
          registerForm.reset();
          showToast(`Welcome, ${resp.user.full_name}! Account created.`, "success");
          checkAuthStatus();
        } else {
          registerAlert.className = "modal-alert";
          registerAlert.textContent = resp.message || "Registration failed.";
          registerAlert.style.display = "block";
        }
      } catch (err) {
        registerAlert.className = "modal-alert";
        if (window.location.protocol === "file:") {
          registerAlert.innerHTML = "<strong>Opened as local file (file://):</strong><br>To create an account and save to the database, run <code>python app.py</code> in PowerShell and open <a href='http://127.0.0.1:5000' target='_blank' style='color:#34d399;font-weight:bold;'>http://127.0.0.1:5000</a>.";
        } else {
          registerAlert.innerHTML = "<strong>Server Unreachable:</strong><br>Please make sure <code>python app.py</code> is running in PowerShell on port 5000.";
        }
        registerAlert.style.display = "block";
      }
    });
  }

  // Login Form Submission
  if (loginForm) {
    loginForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      loginAlert.style.display = "none";
      const payload = Object.fromEntries(new FormData(loginForm).entries());

      try {
        const res = await fetch("/api/login", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        const resp = await res.json();
        if (res.ok) {
          closeModal(loginModal);
          loginForm.reset();
          showToast(`Welcome back, ${resp.user.full_name}!`, "success");
          checkAuthStatus();
        } else {
          loginAlert.className = "modal-alert";
          loginAlert.textContent = resp.message || "Invalid credentials.";
          loginAlert.style.display = "block";
        }
      } catch (err) {
        loginAlert.className = "modal-alert";
        if (window.location.protocol === "file:") {
          loginAlert.innerHTML = "<strong>Opened as local file (file://):</strong><br>To log in and access the database, run <code>python app.py</code> in PowerShell and open <a href='http://127.0.0.1:5000' target='_blank' style='color:#34d399;font-weight:bold;'>http://127.0.0.1:5000</a>.";
        } else {
          loginAlert.innerHTML = "<strong>Server Unreachable:</strong><br>Please make sure <code>python app.py</code> is running in PowerShell on port 5000.";
        }
        loginAlert.style.display = "block";
      }
    });
  }

  // Logout Handler
  if (logoutBtn) {
    logoutBtn.addEventListener("click", async () => {
      try {
        await fetch("/api/logout", { method: "POST" });
        showToast("Logged out successfully.", "success");
        checkAuthStatus();
      } catch {
        showToast("Error during logout.", "error");
      }
    });
  }

  // ---------------------------------------------------------------------------
  // User Dashboard History Operations
  // ---------------------------------------------------------------------------

  async function loadUserDashboard() {
    if (!userAppsTbody) return;
    try {
      const res = await fetch("/api/applications");
      const resp = await res.json();
      if (!res.ok || !resp.applications) return;

      const apps = resp.applications;
      userTotalApps.textContent = apps.length;

      let approvedCount = 0;
      let rejectedCount = 0;

      userAppsTbody.replaceChildren();

      if (!apps.length) {
        userAppsTbody.innerHTML = '<tr><td colspan="9" style="text-align: center; padding: 24px; color: var(--text-muted);">No applications submitted yet. Use the screener below to test your first profile.</td></tr>';
        userApprovedApps.textContent = "0";
        userRejectedApps.textContent = "0";
        return;
      }

      apps.forEach((app) => {
        const preds = parsePredictionsSummary(app.predictions_summary);
        const rfPred = preds["random_forest"]?.prediction || "--";
        const dtPred = preds["decision_tree"]?.prediction || "--";
        const lrPred = preds["logistic_regression"]?.prediction || "--";

        // Consensus logic
        const votes = [rfPred, dtPred, lrPred].filter((p) => p === "Approved").length;
        const consensus = votes >= 2 ? "Approved" : "Rejected";
        if (consensus === "Approved") approvedCount += 1;
        else rejectedCount += 1;

        const row = document.createElement("tr");
        row.innerHTML = `
          <td>#${app.id}</td>
          <td>${formatDate(app.created_at)}</td>
          <td>${formatCurrency(app.loan_amount)}</td>
          <td><span class="field-badge" style="background: rgba(255,255,255,0.06);">${app.cibil_score}</span></td>
          <td><span class="outcome-pill ${rfPred.toLowerCase()}">${rfPred}</span></td>
          <td><span class="outcome-pill ${dtPred.toLowerCase()}">${dtPred}</span></td>
          <td><span class="outcome-pill ${lrPred.toLowerCase()}">${lrPred}</span></td>
          <td><span class="outcome-pill ${consensus.toLowerCase()}">${consensus}</span></td>
          <td><button type="button" class="btn-table-action" data-app-id="${app.id}">View</button></td>
        `;
        userAppsTbody.appendChild(row);
      });

      userApprovedApps.textContent = approvedCount;
      userRejectedApps.textContent = rejectedCount;

      // Add detail click listeners
      userAppsTbody.querySelectorAll(".btn-table-action").forEach((b) => {
        b.addEventListener("click", () => openApplicationDetail(b.dataset.appId));
      });
    } catch {
      userAppsTbody.innerHTML = '<tr><td colspan="9" style="text-align: center; color: var(--rose-main);">Error loading applications.</td></tr>';
    }
  }

  function parsePredictionsSummary(summaryStr) {
    if (!summaryStr) return {};
    const map = {};
    summaryStr.split(";").forEach((chunk) => {
      const parts = chunk.split(":");
      if (parts.length >= 2) {
        map[parts[0]] = {
          prediction: parts[1],
          probability: Number(parts[2] || 0)
        };
      }
    });
    return map;
  }

  function formatDate(dtStr) {
    if (!dtStr) return "--";
    const d = new Date(dtStr);
    return isNaN(d.getTime()) ? dtStr : d.toLocaleDateString("en-IN", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
  }

  async function openApplicationDetail(appId) {
    openModal(detailModal);
    detailModalBody.innerHTML = '<div class="loading-state"><div class="spinner"></div><p>Fetching application details…</p></div>';

    try {
      const res = await fetch(`/api/applications/${appId}`);
      const resp = await res.json();
      if (!res.ok) {
        detailModalBody.innerHTML = `<div class="modal-alert">${resp.message || "Could not load record."}</div>`;
        return;
      }

      const app = resp.application;
      const preds = resp.predictions;

      let predsHtml = '<div class="models-grid" style="margin-top: 14px;">';
      preds.forEach((p) => {
        const meta = modelMeta[p.model_name] || { name: p.model_name, badge: "Model" };
        const isApp = p.predicted_class === "Approved";
        const prob = p.approved_probability ? (p.approved_probability * 100).toFixed(1) : "--";
        predsHtml += `
          <div class="model-result-row">
            <div class="model-row-top">
              <span class="model-title">${meta.name}</span>
              <span class="outcome-pill ${isApp ? "approved" : "rejected"}">${p.predicted_class}</span>
            </div>
            <div class="model-row-meter">
              <div class="prob-meter-track">
                <div class="prob-meter-fill ${isApp ? "approved" : "rejected"}" style="width: ${prob !== "--" ? prob : 50}%"></div>
              </div>
              <span class="prob-label">${prob !== "--" ? prob + "%" : ""}</span>
            </div>
          </div>
        `;
      });
      predsHtml += "</div>";

      detailModalBody.innerHTML = `
        <div class="detail-grid">
          <div class="detail-item"><span class="detail-item-label">Application ID</span><span class="detail-item-val">#${app.id}</span></div>
          <div class="detail-item"><span class="detail-item-label">Date Submitted</span><span class="detail-item-val">${formatDate(app.created_at)}</span></div>
          <div class="detail-item"><span class="detail-item-label">Dependents</span><span class="detail-item-val">${app.no_of_dependents}</span></div>
          <div class="detail-item"><span class="detail-item-label">Education</span><span class="detail-item-val">${app.education}</span></div>
          <div class="detail-item"><span class="detail-item-label">Self-Employed</span><span class="detail-item-val">${app.self_employed}</span></div>
          <div class="detail-item"><span class="detail-item-label">Annual Income</span><span class="detail-item-val">${formatCurrency(app.income_annum)}</span></div>
          <div class="detail-item"><span class="detail-item-label">Loan Amount</span><span class="detail-item-val">${formatCurrency(app.loan_amount)}</span></div>
          <div class="detail-item"><span class="detail-item-label">Loan Term</span><span class="detail-item-val">${app.loan_term} Years</span></div>
          <div class="detail-item"><span class="detail-item-label">CIBIL Score</span><span class="detail-item-val">${app.cibil_score}</span></div>
          <div class="detail-item"><span class="detail-item-label">Residential Assets</span><span class="detail-item-val">${formatCurrency(app.residential_assets_value)}</span></div>
          <div class="detail-item"><span class="detail-item-label">Commercial Assets</span><span class="detail-item-val">${formatCurrency(app.commercial_assets_value)}</span></div>
          <div class="detail-item"><span class="detail-item-label">Luxury Assets</span><span class="detail-item-val">${formatCurrency(app.luxury_assets_value)}</span></div>
          <div class="detail-item"><span class="detail-item-label">Bank Balance</span><span class="detail-item-val">${formatCurrency(app.bank_asset_value)}</span></div>
        </div>
        <h4 class="detail-preds-title">Model Decisions for this Applicant</h4>
        ${predsHtml}
      `;
    } catch {
      detailModalBody.innerHTML = '<div class="modal-alert">Error fetching application details.</div>';
    }
  }

  // ---------------------------------------------------------------------------
  // Admin Portal Operations
  // ---------------------------------------------------------------------------

  async function loadAdminPortal() {
    if (!adminSection) return;
    adminSection.style.display = "block";

    try {
      const res = await fetch("/api/admin/stats");
      const resp = await res.json();
      if (!res.ok) return;

      const stats = resp.stats;
      adminUserCount.textContent = stats.total_users;
      adminAppCount.textContent = stats.total_applications;
      adminDbStatus.textContent = "CONNECTED";
      adminDbName.textContent = `${(stats.database_engine || "MySQL").toUpperCase()} Storage`;

      // Render prediction counts
      adminPredCountsTbody.replaceChildren();
      stats.prediction_counts.forEach((item) => {
        const row = document.createElement("tr");
        const meta = modelMeta[item.model_name] || { name: item.model_name };
        row.innerHTML = `
          <td><strong>${meta.name}</strong></td>
          <td><span class="outcome-pill ${item.predicted_class.toLowerCase()}">${item.predicted_class}</span></td>
          <td>${item.count}</td>
        `;
        adminPredCountsTbody.appendChild(row);
      });

      loadAdminApplications();
    } catch (e) {
      console.warn("Could not load admin stats:", e);
    }
  }

  async function loadAdminApplications(query = "") {
    if (!adminAppsTbody) return;
    try {
      const url = query ? `/api/admin/applications?q=${encodeURIComponent(query)}` : "/api/admin/applications";
      const res = await fetch(url);
      const resp = await res.json();
      if (!res.ok || !resp.applications) return;

      adminAppsTbody.replaceChildren();
      if (!resp.applications.length) {
        adminAppsTbody.innerHTML = '<tr><td colspan="9" style="text-align: center; color: var(--text-muted);">No applications found matching search.</td></tr>';
        return;
      }

      resp.applications.forEach((app) => {
        const row = document.createElement("tr");
        row.innerHTML = `
          <td>#${app.id}</td>
          <td>${app.full_name}</td>
          <td>${app.email}</td>
          <td>${formatCurrency(app.income_annum)}</td>
          <td>${formatCurrency(app.loan_amount)}</td>
          <td>${app.loan_term}y</td>
          <td>${app.cibil_score}</td>
          <td>${formatDate(app.created_at)}</td>
          <td><button type="button" class="btn-table-action" data-app-id="${app.id}">Inspect</button></td>
        `;
        adminAppsTbody.appendChild(row);
      });

      adminAppsTbody.querySelectorAll(".btn-table-action").forEach((b) => {
        b.addEventListener("click", () => openApplicationDetail(b.dataset.appId));
      });
    } catch {
      adminAppsTbody.innerHTML = '<tr><td colspan="9">Error loading applications.</td></tr>';
    }
  }

  if (adminSearchBtn && adminSearchInput) {
    adminSearchBtn.addEventListener("click", () => {
      loadAdminApplications(adminSearchInput.value.trim());
    });
    adminSearchInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") loadAdminApplications(adminSearchInput.value.trim());
    });
  }

  // ---------------------------------------------------------------------------
  // Screener Form & Real-Time CIBIL / Collateral Inputs
  // ---------------------------------------------------------------------------

  function updateCibilBadge(val) {
    const score = Number(val) || 300;
    const badge = document.getElementById("cibil-score-badge");
    const fill = document.getElementById("cibil-meter-fill");
    if (!badge || !fill) return;

    let text = "Poor (<550)";
    let color = "var(--rose-main)";
    let bg = "var(--rose-wash)";
    let border = "var(--rose-border)";

    if (score >= 750) {
      text = "Excellent (750+)";
      color = "var(--emerald-light)";
      bg = "var(--emerald-wash)";
      border = "var(--emerald-border)";
    } else if (score >= 650) {
      text = "Good (650-749)";
      color = "var(--indigo-light)";
      bg = "var(--indigo-wash)";
      border = "rgba(99, 102, 241, 0.3)";
    } else if (score >= 550) {
      text = "Fair (550-649)";
      color = "var(--amber-main)";
      bg = "var(--amber-wash)";
      border = "var(--amber-border)";
    }

    badge.textContent = `${score} · ${text}`;
    badge.style.color = color;
    badge.style.background = bg;
    badge.style.border = `1px solid ${border}`;

    const pct = Math.max(0, Math.min(100, ((score - 300) / 600) * 100));
    fill.style.width = `${pct}%`;
  }

  function updateCollateral() {
    const res = Number(document.getElementById("residential_assets_value")?.value) || 0;
    const comm = Number(document.getElementById("commercial_assets_value")?.value) || 0;
    const lux = Number(document.getElementById("luxury_assets_value")?.value) || 0;
    const bank = Number(document.getElementById("bank_asset_value")?.value) || 0;
    const loan = Number(document.getElementById("loan_amount")?.value) || 1;

    const totalAssets = res + comm + lux + bank;
    const ratio = totalAssets / loan;

    if (totalAssetsVal) totalAssetsVal.textContent = formatCurrency(totalAssets);
    if (loanAmountDisp) loanAmountDisp.textContent = formatCurrency(loan);

    if (collateralRatioVal && collateralBar) {
      collateralRatioVal.textContent = `${ratio.toFixed(2)}x Coverage`;
      const barPct = Math.min(100, (ratio / 3) * 100);
      collateralBar.style.width = `${barPct}%`;

      if (ratio >= 1.5) collateralRatioVal.style.color = "var(--emerald-light)";
      else if (ratio >= 1.0) collateralRatioVal.style.color = "var(--indigo-light)";
      else collateralRatioVal.style.color = "var(--amber-main)";
    }
  }

  function makeControl(feature) {
    const isCibil = feature === "cibil_score";
    const wrapper = document.createElement("div");
    wrapper.className = isCibil ? "field cibil-highlight-field" : "field";

    const labelRow = document.createElement("div");
    labelRow.className = "field-label-row";

    const label = document.createElement("label");
    label.className = "field-label";
    label.htmlFor = feature;
    label.textContent = featureLabels[feature] || feature.replaceAll("_", " ");
    labelRow.appendChild(label);

    if (isCibil) {
      const badge = document.createElement("span");
      badge.className = "field-badge";
      badge.id = "cibil-score-badge";
      badge.textContent = "600 · Fair";
      labelRow.appendChild(badge);
    }
    wrapper.appendChild(labelRow);

    if (numericSet.has(feature)) {
      const input = document.createElement("input");
      input.type = "number";
      input.id = feature;
      input.name = feature;
      input.required = true;
      input.value = metadata.numeric_defaults[feature] || 0;

      if (isCibil) {
        input.min = "300";
        input.max = "900";
        input.step = "5";
        input.addEventListener("input", (e) => updateCibilBadge(e.target.value));

        const meterTrack = document.createElement("div");
        meterTrack.className = "cibil-meter-track";
        const meterFill = document.createElement("div");
        meterFill.className = "cibil-meter-fill";
        meterFill.id = "cibil-meter-fill";
        meterTrack.appendChild(meterFill);

        wrapper.appendChild(input);
        wrapper.appendChild(meterTrack);
        return wrapper;
      }

      if (feature.includes("asset") || feature.includes("amount") || feature.includes("income")) {
        input.step = "100000";
        input.addEventListener("input", updateCollateral);
      }

      wrapper.appendChild(input);
    } else {
      const select = document.createElement("select");
      select.id = feature;
      select.name = feature;
      select.required = true;

      const cats = metadata.categorical_values[feature] || [];
      cats.forEach((cat) => {
        const cleanCat = String(cat).trim();
        const option = document.createElement("option");
        option.value = cleanCat;
        option.textContent = cleanCat;
        select.appendChild(option);
      });
      wrapper.appendChild(select);
    }

    return wrapper;
  }

  function renderForm() {
    if (!fieldsHost) return;
    fieldsHost.replaceChildren();
    const available = new Set(metadata.features);

    formGroups.forEach((grp) => {
      const validFeatures = grp.features.filter((f) => available.has(f));
      if (!validFeatures.length) return;

      const groupCard = document.createElement("div");
      groupCard.className = "form-group-card";

      const title = document.createElement("div");
      title.className = "form-group-title";
      title.textContent = grp.title;
      groupCard.appendChild(title);

      validFeatures.forEach((f) => groupCard.appendChild(makeControl(f)));
      fieldsHost.appendChild(groupCard);
    });

    const cibilInput = document.getElementById("cibil_score");
    if (cibilInput) updateCibilBadge(cibilInput.value);
    updateCollateral();
  }

  function renderMetrics() {
    if (!metricsHost) return;
    metricsHost.replaceChildren();
    const metricsList = data ? data.metrics : [];

    metricsList.forEach((metric) => {
      const row = document.createElement("tr");
      const isChamp = metric.model === "random_forest";
      if (isChamp) row.className = "champion-row";

      const cells = [
        modelMeta[metric.model]?.name || metric.model,
        `${(metric.accuracy * 100).toFixed(2)}%`,
        `${(metric.precision_approved * 100).toFixed(2)}%`,
        `${(metric.recall_approved * 100).toFixed(2)}%`,
        `${(metric.f1_approved * 100).toFixed(2)}%`,
        metric.roc_auc ? `${(metric.roc_auc * 100).toFixed(2)}%` : "--",
      ];

      cells.forEach((val) => {
        const td = document.createElement("td");
        td.textContent = val;
        row.appendChild(td);
      });

      metricsHost.appendChild(row);
    });
  }

  // ---------------------------------------------------------------------------
  // Execution of Predictions (Flask API Backend with Offline Fallback)
  // ---------------------------------------------------------------------------

  async function executePredictions(input) {
    const submitBtn = document.getElementById("submit-btn");
    const submitText = document.getElementById("submit-btn-text");
    const submitSpinner = document.getElementById("submit-spinner");
    const submitArrow = document.getElementById("submit-btn-arrow");

    if (submitSpinner) submitSpinner.style.display = "inline-block";
    if (submitArrow) submitArrow.style.display = "none";
    if (submitText) submitText.textContent = "Analyzing Multi-Model Outputs…";

    try {
      // 1. Attempt Flask Backend API
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(input)
      });

      if (res.ok) {
        const resp = await res.json();
        renderPredictionCards(resp.predictions, resp.consensus, resp.agreement);
        updateDriverInsights(input, resp.consensus === "Approved");

        if (resp.application_id) {
          showToast(`✓ Application #${resp.application_id} saved to your account!`, "success");
          loadUserDashboard();
        }
        return;
      }
    } catch {
      // Fallback to client-side predictor if Flask server isn't serving API
    } finally {
      if (submitSpinner) submitSpinner.style.display = "none";
      if (submitArrow) submitArrow.style.display = "inline-block";
      if (submitText) submitText.textContent = "Run All 3 Classifiers";
      if (postPredActions) postPredActions.style.display = "flex";
    }

    // Client-side offline fallback
    if (window.LoanPredictor && data && data.models) {
      const clientPredictions = {};
      Object.entries(data.models).forEach(([key, m]) => {
        const pred = window.LoanPredictor.predict(metadata, m, input);
        const appIdx = m.classes.indexOf(metadata.positive_class);
        clientPredictions[key] = {
          prediction: pred.outcome,
          approval_probability: pred.probabilities[appIdx]
        };
      });
      const votes = Object.values(clientPredictions).filter((p) => p.prediction === metadata.positive_class).length;
      const consensus = votes >= 2 ? "Approved" : "Rejected";
      renderPredictionCards(clientPredictions, consensus, `${votes}/3 models recommend ${consensus}`);
      updateDriverInsights(input, consensus === "Approved");
    }
  }

  function renderPredictionCards(predictions, consensus, agreementText) {
    resultsHost.replaceChildren();

    const isConsensusApproved = consensus === "Approved";
    consensusBanner.className = `consensus-banner ${isConsensusApproved ? "approved" : "rejected"}`;
    consensusIcon.textContent = isConsensusApproved ? "✅" : "⚠️";
    consensusStatus.textContent = isConsensusApproved ? "RECOMMEND APPROVAL" : "RECOMMEND REJECTION";
    consensusDetail.textContent = agreementText || `Consensus: ${isConsensusApproved ? "Approved" : "Rejected"}`;

    const order = ["random_forest", "decision_tree", "logistic_regression"];

    order.forEach((key) => {
      const p = predictions[key];
      if (!p) return;

      const meta = modelMeta[key] || { name: key, badge: "Model", isChampion: false, note: "" };
      const isApproved = p.prediction === "Approved";
      const prob = p.approval_probability !== undefined ? (p.approval_probability * 100).toFixed(1) : "--";

      const row = document.createElement("div");
      row.className = `model-result-row ${meta.isChampion ? "champion" : ""}`;

      row.innerHTML = `
        <div class="model-row-top">
          <div class="model-name-group">
            <span class="model-title">${meta.name}</span>
            <span class="model-badge-tag">${meta.badge}</span>
          </div>
          <span class="outcome-pill ${isApproved ? "approved" : "rejected"}">${isApproved ? "✓ Approved" : "✕ Rejected"}</span>
        </div>
        <div class="model-row-meter">
          <div class="prob-meter-track">
            <div class="prob-meter-fill ${isApproved ? "approved" : "rejected"}" style="width: 0%;"></div>
          </div>
          <span class="prob-label">${prob !== "--" ? prob + "%" : ""}</span>
        </div>
        <div class="model-arch-note">${meta.note}</div>
      `;

      resultsHost.appendChild(row);

      window.requestAnimationFrame(() => {
        const fill = row.querySelector(".prob-meter-fill");
        if (fill && prob !== "--") fill.style.width = `${prob}%`;
      });
    });
  }

  function updateDriverInsights(input, isApproved) {
    if (!driverCard || !driverList) return;
    driverCard.style.display = "block";
    driverList.replaceChildren();

    const cibil = Number(input.cibil_score);
    const loan = Number(input.loan_amount);
    const income = Number(input.income_annum);
    const totalAssets =
      Number(input.residential_assets_value || 0) +
      Number(input.commercial_assets_value || 0) +
      Number(input.luxury_assets_value || 0) +
      Number(input.bank_asset_value || 0);

    const coverage = totalAssets / (loan || 1);
    const factors = [];

    if (cibil >= 750) {
      factors.push(`CIBIL score of ${cibil} is in the top credit tier (750+), indicating strong historic repayment.`);
    } else if (cibil >= 650) {
      factors.push(`CIBIL score of ${cibil} meets institutional prime lending benchmarks.`);
    } else if (cibil >= 550) {
      factors.push(`CIBIL score of ${cibil} is moderate (550-649), requiring additional collateral security.`);
    } else {
      factors.push(`CIBIL score of ${cibil} falls below the 550 threshold, driving primary rejection consensus.`);
    }

    if (coverage >= 2.0) {
      factors.push(`Collateral coverage is robust at ${coverage.toFixed(2)}x the loan request, mitigating default risk.`);
    } else if (coverage >= 1.0) {
      factors.push(`Collateral coverage is adequate at ${coverage.toFixed(2)}x the requested borrowing.`);
    } else {
      factors.push(`Total declared asset valuations do not fully cover the loan amount (${coverage.toFixed(2)}x coverage).`);
    }

    if (loan > income * 4) {
      factors.push(`Loan-to-income ratio exceeds 4x annual earnings, indicating elevated debt servicing risk.`);
    } else {
      factors.push(`Requested loan amount is well-balanced relative to declared annual earnings of ${formatCurrency(income)}.`);
    }

    factors.forEach((f) => {
      const li = document.createElement("li");
      li.textContent = f;
      driverList.appendChild(li);
    });
  }

  function applyPreset(presetKey) {
    const preset = presets[presetKey];
    if (!preset) return;

    Object.entries(preset).forEach(([key, val]) => {
      const el = document.getElementById(key);
      if (el) el.value = val;
    });

    const cibilInput = document.getElementById("cibil_score");
    if (cibilInput) updateCibilBadge(cibilInput.value);
    updateCollateral();

    const currentValues = Object.fromEntries(new FormData(form).entries());
    executePredictions(currentValues);
  }

  // ---------------------------------------------------------------------------
  // Visual Chart Tabs & Actions
  // ---------------------------------------------------------------------------

  const tabs = document.querySelectorAll(".visual-tab");
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");

      const chartType = tab.dataset.chart;
      document.querySelectorAll(".chart-view").forEach((view) => view.classList.remove("active"));
      const targetView = document.getElementById(`chart-${chartType}-view`);
      if (targetView) targetView.classList.add("active");
    });
  });

  if (presetContainer) {
    presetContainer.addEventListener("click", (e) => {
      const chip = e.target.closest(".preset-chip");
      if (chip && chip.dataset.preset) applyPreset(chip.dataset.preset);
    });
  }

  if (fillDefaultsBtn) {
    fillDefaultsBtn.addEventListener("click", () => {
      Object.entries(metadata.numeric_defaults).forEach(([f, val]) => {
        const el = document.getElementById(f);
        if (el) el.value = val;
      });
      const cibilInput = document.getElementById("cibil_score");
      if (cibilInput) updateCibilBadge(cibilInput.value);
      updateCollateral();

      const currentValues = Object.fromEntries(new FormData(form).entries());
      executePredictions(currentValues);
    });
  }

  if (printReportBtn) {
    printReportBtn.addEventListener("click", () => window.print());
  }

  if (scrollToHistoryBtn) {
    scrollToHistoryBtn.addEventListener("click", () => {
      if (currentUser && userDashboardSection) {
        userDashboardSection.scrollIntoView({ behavior: "smooth" });
      } else {
        openModal(loginModal);
      }
    });
  }

  if (newAppBtn) {
    newAppBtn.addEventListener("click", () => {
      form.scrollIntoView({ behavior: "smooth" });
    });
  }

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    if (!form.reportValidity()) return;
    const values = Object.fromEntries(new FormData(form).entries());
    executePredictions(values);
    document.getElementById("prediction-results").scrollIntoView({ behavior: "smooth", block: "nearest" });
  });

  form.addEventListener("reset", () => {
    window.setTimeout(() => {
      resultsHost.replaceChildren();
      resultsHost.innerHTML = '<div class="results-empty-state"><p>Form reset. Select a preset or enter values to evaluate loan approval.</p></div>';
      consensusBanner.className = "consensus-banner";
      consensusIcon.textContent = "⚖️";
      consensusStatus.textContent = "Awaiting Input";
      consensusDetail.textContent = "Submit the applicant profile to generate model decisions.";
      if (driverCard) driverCard.style.display = "none";
      if (postPredActions) postPredActions.style.display = "none";
      const cibilInput = document.getElementById("cibil_score");
      if (cibilInput) updateCibilBadge(cibilInput.value);
      updateCollateral();
    }, 0);
  });

  // ---------------------------------------------------------------------------
  // App Bootstrapping
  // ---------------------------------------------------------------------------

  renderForm();
  renderMetrics();
  checkAuthStatus();
  applyPreset("prime");
}());
