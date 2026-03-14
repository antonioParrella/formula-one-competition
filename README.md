## Prerequisites

Before running the setup scripts, ensure you have the following installed:
- **Python 3.8 or higher**: [Download here](https://www.python.org/downloads/)
- **pip**: Usually included with Python installations.[[2](https://www.google.com/url?sa=E&q=https%3A%2F%2Fvertexaisearch.cloud.google.com%2Fgrounding-api-redirect%2FAUZIYQFp9x0mMthHyamvsSO-SeEezBLMfCspkGUgyhcof6u4KwmjRkqs_9DSY0Uv1FDo6KVURn4a5_FMDlQG9qIQW6U7IASGHQYXFN9unweTpUpae44-8x82XyIPALcC88m8TTwQ678t0rK_u8UVs52YZxS_LVWoh4a6VD1nsLB_0-Zowl4O__eQB5V97E1WNrv_O6wQ1Jf89QIEs1YbhZG797o1eMphCJiHC6CoXAOc4FZHrZTjhj5av7ddHr767MIV)][[3](https://www.google.com/url?sa=E&q=https%3A%2F%2Fvertexaisearch.cloud.google.com%2Fgrounding-api-redirect%2FAUZIYQHD_-RTo6Xz_sGOU3WRIk-BWVQtrUIz9Gma0Z9bxkRhamCDrjQzi3jpK1kzrOC_NFtBEs77g7ulyrM3VduZxzHnzIvoUBVkCgMBnjgjUYnb68xp72662F6crZTkOamnPTBmsnc%3D)]

---

## 🛠️ Installation & Setup

We have provided automated scripts to handle the environment creation and dependency installation for you.

### 🪟 Windows
1. Locate the `setup.bat` file in the project folder.
2. **Double-click** `setup.bat`.
3. The script will:
   - Create a virtual environment (`venv`).
   - Upgrade `pip`.
   - Install all requirements from `requirements.txt`.
4. Keep the terminal window open to run the app.

### 🍎 macOS / Linux
1. Open your **Terminal** and navigate to the project folder.
2. Make the setup script executable (first time only):
   ```bash
   chmod +x setup.sh