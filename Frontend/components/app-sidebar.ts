import { LitElement, html, css } from "lit";
import { customElement, property, state } from "lit/decorators.js";

@customElement("app-sidebar")
export class AppSidebar extends LitElement {
  static styles = css`
    :host {
      display: block;
      height: 100vh;
    }

    .sidebar {
      margin: 0;
      padding: 0;
      height: 100%;
      background-color: #1e3a52;
      width: 250px;
      transition: width 0.3s ease;
      overflow: hidden;
      position: relative;
      display: flex;
      flex-direction: column;
    }

    .sidebar.collapsed {
      width: 50px;
    }

    button {
      position: absolute;
      top: 20px;
      right: 10px;
      width: 30px;
      height: 30px;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      background: #2c5f7c;
      color: white;
      font-size: 16px;
      transition: transform 0.2s;
      z-index: 1;
    }

    button:hover {
      background: #3d7a9a;
    }

    .sidebar-content {
      padding: 0px;
      overflow-y: auto;
      overflow-x: hidden;
      flex: 1;
      scrollbar-width: thin;
      scrollbar-color: #2c5f7c #1e3a52;
    }

    .sidebar-content::-webkit-scrollbar {
      width: 6px;
    }

    .sidebar-content::-webkit-scrollbar-track {
      background: #1e3a52;
    }

    .sidebar-content::-webkit-scrollbar-thumb {
      background-color: #2c5f7c;
      border-radius: 3px;
    }

    .sidebar-content::-webkit-scrollbar-thumb:hover {
      background-color: #3d7a9a;
    }

    .sidebar.collapsed .sidebar-content {
      display: none;
      transition: display 0.3s ease;
    }

    .sidebar-title {
      color: white;
      font-size: 18px;
      font-weight: bold;
      margin-left: 20px;
      margin-top: 10px;
    }
    
    .sidebar-files {
      margin-top: 10px;
      color: #ccc;
    }
  `;

  @property({ type: String })
  apiUrl = "http://localhost:8000";

  @state()
  private collapsed = false;

  @state()
  private all_files: { name: string; path: string }[] = [];

  private toggleSidebar() {
    this.collapsed = !this.collapsed;
  }

  connectedCallback() {
    super.connectedCallback();
    window.addEventListener('upload-complete', this._handleUpload as EventListener);
    this.loadExistingFiles();
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    window.removeEventListener('upload-complete', this._handleUpload as EventListener);
  }

  private async loadExistingFiles() {
    try {
      const response = await fetch(`${this.apiUrl}/files/`);
      if (!response.ok) throw new Error(`HTTP error: ${response.status}`);

      const data = await response.json();
      console.log('📂 Loaded files from server:', data);

      this.all_files = data.files.map((file: any) => ({
        name: file.filename,
        path: `uploads/${file.filename}`
      }));
    } catch (error) {
      console.error('❌ Failed to load files:', error);
    }
  }

  private _handleUpload = (e: CustomEvent) => {
    console.log('📂 Sidebar received upload event:', e.detail);
    this.addUploadedFiles(e.detail.results);
  };

  addUploadedFiles(uploadResults: any[]) {
    console.log('📂 Adding files:', uploadResults);
    const newFiles = uploadResults.map(result => ({
      name: result.filename,
      path: `uploads/${result.filename}`
    }));
    this.all_files = [...this.all_files, ...newFiles];
  }
  
  
  render() {
    return html`
      <div class="sidebar ${this.collapsed ? "collapsed" : ""}">
        <button @click=${this.toggleSidebar}>
          ${this.collapsed ? "›" : "‹"}
        </button>
        <div class="sidebar-content">
          <div class="sidebar-title">
            <h3>Files</h3>
          </div>
          <div class="sidebar-files">
            ${this.all_files.length === 0
              ? html`<p style="padding: 0 20px; color: #888;">No files uploaded</p>`
              : this.all_files.map(file => html`
                  <sidebar-file name="${file.name}" path="${file.path}"></sidebar-file>
                `)
            }
          </div>
        </div>
      </div>
    `;
  }
}
