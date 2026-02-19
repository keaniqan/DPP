import { LitElement, html, css } from "lit";
import { customElement, query, state } from "lit/decorators.js";
    
import "./components/heading1";
import "./components/app-layout";
import './components/app-sidebar';
import './components/file-upload';
import './components/sidebar-file';
import './components/popover-menu';
import './components/results';
import { AppSidebar } from "./components/app-sidebar";

@customElement("app-main")
export class AppMain extends LitElement {
  @query('app-sidebar')
  sidebar!: AppSidebar;

  @state()
  private _selectedFile = localStorage.getItem("selectedFile") ?? "";

  private handleUploadComplete(e: CustomEvent) {
    const { results } = e.detail;
    this.sidebar.addUploadedFiles(results);
  }

  private handleFileSelected(e: CustomEvent) {
    this._selectedFile = e.detail.name;
    localStorage.setItem("selectedFile", e.detail.name);
  }

  render() {
    return html`
    <app-layout @file-selected=${this.handleFileSelected}>
        <heading-1 text="Evaluation Analysis"></heading-1>
        <file-upload
            upload-url="http://localhost:8000/upload/"
            @upload-complete=${this.handleUploadComplete}
        ></file-upload>
        <br>
        <results-component
            .selectedFile=${this._selectedFile}
        ></results-component>
    </app-layout>
    `;
  }
}