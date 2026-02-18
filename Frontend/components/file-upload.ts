import {html, css, LitElement} from 'lit';
import {customElement, property, state} from 'lit/decorators.js';

@customElement('file-upload')
export class FileUpload extends LitElement {
  static styles = css`
    :host {
      display: block;
    }

    .drop-zone {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        border: 2px dashed #cccccc;
        border-radius: 8px;
        padding: 40px;
        background-color: #f9f9f9;
        cursor: pointer;
        text-align: center;
    }

    .drop-zone svg {
        margin: 0 auto;
    }

    .drop-zone p {
        font-size: 16px;
        color: #666;
    }

    .drop-zone a {
        color: #2196F3;
        text-decoration: underline;
        cursor: pointer;
    }

    .upload-container {
      border: 2px dashed #ffffff4d;
      border-radius: 12px;
      padding: 25px 40px;
      padding-top: 15px;
      text-align: center;
      background: #33495c5c;
      transition: all 0.3s ease;
    }

    .upload-container.drag-over {
      border-color: #3d7a9a;
      background: #2c5f7c;
    }

    .upload-icon {
      width: 64px;
      height: 64px;
      margin: 0 auto 24px;
    }

    .upload-icon svg {
      width: 100%;
      height: 100%;
      fill: none;
      stroke: #e8eaed;
      stroke-width: 2;
      stroke-linecap: round;
      stroke-linejoin: round;
    }

    .upload-text {
      font-size: 15px;
      color: #000000a5;
      margin-bottom: 28px;
      margin-top: 28px;
      font-weight: 400;
    }

    .upload-subtext {
      font-size: 14px;
      color: #797979;
      margin-bottom: 8px;
    }

    .browse-button {
      background: #253c4982;
      color: white;
      border: none;
      border-radius: 4px;
      margin-top: 16px;
      padding: 12px 48px;
      font-size: 16px;
      font-weight: 500;
      cursor: pointer;
      transition: background 0.2s ease;
    }

    .browse-button:hover {
      background: #3d7a9a;
    }

    .browse-button:active {
      background: #5a7f95;
    }

    input[type="file"] {
      display: none;
    }

    .file-list {
      margin-top: 20px;
      text-align: left;
    }

    .file-item {
      padding: 8px 12px;
      background: #2c5f7c;
      border-radius: 6px;
      margin-bottom: 8px;
      font-size: 14px;
      color: #e8eaed;
    }
  `;

  @property({type: String})
  acceptedFormats = ['txt', 'md'];

  @property({type: String})
  uploadUrl = 'http://localhost:8000/upload/';

  @state()
  private dragOver = false;

  @state()
  private files: File[] = [];

  private handleDragOver(e: DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    this.dragOver = true;
  }

  private handleDragLeave(e: DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    this.dragOver = false;
  }

  private handleDrop(e: DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    this.dragOver = false;

    const droppedFiles = Array.from(e.dataTransfer?.files || []);
    this.processFiles(droppedFiles);
  }

  private handleBrowseClick() {
    const input = this.shadowRoot?.querySelector('input[type="file"]') as HTMLInputElement;
    input?.click();
  }

  private handleFileSelect(e: Event) {
    const input = e.target as HTMLInputElement;
    const selectedFiles = Array.from(input.files || []);
    this.processFiles(selectedFiles);
  }

  private processFiles(newFiles: File[]) {
    const validFiles = newFiles.filter(file => {
      const extension = file.name.split('.').pop()?.toLowerCase();
      return extension && this.acceptedFormats.includes(extension);
    });

    this.files = [...this.files, ...validFiles];

    // Dispatch custom event with files
    this.dispatchEvent(new CustomEvent('files-selected', {
      detail: { files: this.files },
      bubbles: true,
      composed: true
    }));
  }

  @state()
  private uploading = false;

  @state()
  private uploadResults: any[] = [];

  private async uploadFiles() {
    if (this.files.length === 0) return;

    this.uploading = true;
    const formData = new FormData();

    this.files.forEach(file => {
      formData.append('files', file);
    });

    try {
      const response = await fetch(this.uploadUrl, {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();
      this.uploadResults = result.files;

      this.dispatchEvent(new CustomEvent('upload-complete', {
        detail: { results: result.files },
        bubbles: true,
        composed: true
      }));

      // Clear files after successful upload
      this.files = [];
    } catch (error) {
      console.error('Upload failed:', error);
      this.dispatchEvent(new CustomEvent('upload-error', {
        detail: { error },
        bubbles: true,
        composed: true
      }));
    } finally {
      this.uploading = false;
    }
  }

  render() {
    return html`
      <p class="upload-subtext">Supported formats: ${this.acceptedFormats.join(', ')}</p>
      <div 
         class="drop-zone" ${this.dragOver ? 'drag-over' : ''}"
        @dragover=${this.handleDragOver}
        @dragleave=${this.handleDragLeave}
        @drop=${this.handleDrop}
      >
          <div class="inner-drop-zone">
            <svg xmlns="http://www.w3.org/2000/svg" width="80" height="80" viewBox="0 0 24 24" fill="none" stroke="#cccccc" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 16V8"/>
                <path d="M9 11l3-3 3 3"/>
                <path d="M20 16.58A5 5 0 0 0 18 7h-1.26A8 8 0 1 0 4 15.25"/>
            </svg>
            <p>Drop files here or <a href="#" @click=${this.handleBrowseClick}>browse</a></p>
        </div>

        <input 
          type="file" 
          multiple
          accept="${this.acceptedFormats.map(f => `.${f}`).join(',')}"
          @change=${this.handleFileSelect}
        />
      </div>

      ${this.files.length > 0 ? html`
        <div class="file-list">
          ${this.files.map(file => html`
            <div class="file-item">${file.name}</div>
          `)}
        </div>
        <button 
          class="browse-button" 
          @click=${this.uploadFiles}
          ?disabled=${this.uploading}
          style="margin-top: 16px;"
        >
          ${this.uploading ? 'Uploading...' : 'Upload Files'}
        </button>
      ` : ''}
    `;
  }
}
