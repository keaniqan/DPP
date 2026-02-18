import { LitElement, html, css } from "lit";
import { customElement, query } from "lit/decorators.js";
    
import "./components/heading1";
import "./components/app-layout";
import './components/app-sidebar';
import './components/file-upload';
import './components/sidebar-file';
import './components/popover-menu';
import './components/run-analysis-button';
import './components/results';
import { AppSidebar } from "./components/app-sidebar";

const sampleResults: Array<Record<string, string>> = [
  { metric: "Vocabulary Analysis", value: "150 unique tokens" },
  { metric: "Total Tokens", value: "1000 tokens" },
  { metric: "Vendi Score", value: "0.75" },
  { metric: "Compression Ratio", value: "0.85" },
];

@customElement("app-main")
export class AppMain extends LitElement {
  @query('app-sidebar')
  sidebar!: AppSidebar;

  private handleUploadComplete(e: CustomEvent) {
    const { results } = e.detail;
    this.sidebar.addUploadedFiles(results);
  }

  render() {
    return html`
    <app-layout>
        <heading-1 text="Evaluation Analysis"></heading-1>
        <file-upload
            upload-url="http://localhost:8000/upload/"
            @upload-complete=${this.handleUploadComplete}
        ></file-upload>
        <br>
        <run-analysis-button></run-analysis-button>
        <br>
        <heading-1 text="Analysis Results"></heading-1>
        <results-component .results="${sampleResults}"></results-component>
        <!-- <h1>Text based Analysis</h1>
            <h3>Vocabulary Analysis</h3>
            <h3>Compression Ratio</h3>
            <h3>n-gram score</h3>    
        <h1>Embedding based Analysis</h1>
            <h3>Determinantal Point Process</h3>
            <h3>Vendi Score</h3>
            <h3>NovaAScore</h3>    
        <h1>LLM as Judge based Analysis</h1>
            <h3>Engagement Analysis</h3>
            <h3>Accuracy Analysis</h3>
            <h3>Content Analysis</h3> -->
    </app-layout>
    `;
  }
}