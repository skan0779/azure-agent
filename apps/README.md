

### 1. Create Static Web App

Build Details:
Source = Other
Deployment authorization policy: Deployment Token

### 2. Setting Github Repository Actions

Copy Deployment token:
Azure Portal -> Static Web App -> Manage deployment token

Add New Repository secret:
Settings -> Secrets and Variables -> Actions -> AZURE_STATIC_WEB_APPS_API_TOKEN: paste deploy token
