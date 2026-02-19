import { html, css, LitElement } from "lit";
import { customElement, property, state } from "lit/decorators.js";


@customElement("results-component")
export class ResultsComponent extends LitElement {
  static styles = css`
    :host {
      display: block;
    }

    .results-container {
      background-color: #ffffff;
      border: 1px solid #cccccc;
      border-radius: 8px;
      padding: 16px;
      margin-top: 16px;
    }

    .results-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .export-btn {
      background-color: #438ae0;
      color: white;
      border: dashed #acc3df;
      border-radius: 4px;
      cursor: pointer;
      font-size: 14px;
      width: 150px;
      height: 36px;
      margin-left: 8px;
    }

    .export-btn:hover {
      background-color: #357abd;
    }

    .export-btn:disabled {
      background-color: #a0b9d8;
      cursor: not-allowed;
    }

    .error-msg {
      text-align: center;
      color: #cc0000;
      padding: 8px;
      font-size: 13px;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 8px;
    }

    thead {
      background-color: #f5f5f5;
    }

    th,
    td {
      padding: 10px 12px;
      text-align: left;
      border: 1px solid #dddddd;
    }

    th {
      font-weight: 600;
      color: #333333;
    }

    tbody tr:nth-child(even) {
      background-color: #fafafa;
    }

    tbody tr:hover {
      background-color: #f0f0f0;
    }

    .no-results {
      text-align: center;
      color: #888888;
      padding: 16px;
    }

    .results-container-header {
    display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .matrix-container {
      margin-top: 16px;
      width: 100%;
    }

    #similarity-matrix {
      width: 100%;
      min-height: 400px;
    }
  `;

  @property({ type: Array }) results: Array<Record<string, string>> = [];
  @property({ type: String }) selectedFile = "";
  @state() private _loading = false;
  @state() private _error = "";
  @state() private _evaluations: Record<string, Record<string, unknown>> = {};
  // Watch for selectedFile changes and update table

  updated(changedProps: Map<string, unknown>) {
    if (changedProps.has("selectedFile") && this.selectedFile) {
      this._showFileResults(this.selectedFile);
    }
  }

  private _showFileResults(filename: string) {
    const metrics = this._evaluations[filename];
    if (!metrics) {
      this.results = [];
      this._error = `No evaluation found for "${filename}". Run analysis first.`;
      return;
    }

    this._error = "";
    this.results = Object.entries(metrics).map(([key, value]) => ({
      metric: key,
      value: String(value),
    }));
  }

  render() {
    return html`
      <div class="results-container-header">
        <span style="font-size: 18px; font-weight: 600; color: #333; width: 100%; margin-right: 0px;">Evaluation Results</span>
        <button
            class="export-btn"
            @click=${this._handleButtonClick}
            ?disabled=${this._loading}
            >
        ${this._loading ? "Running..." : "Run Analysis"}
        </button>
      </div>

      <div class="results-container">
        <div class="results-header">
          <span style="font-weight:600; color:#333;">
            ${this.selectedFile ? `📄 ${this.selectedFile}` : "No file selected"}
          </span>

        </div>

        ${this._error
          ? html`<p class="error-msg">${this._error}</p>`
          : ""}

        ${this.results.length > 0
          ? html`
              <table>
                <thead>
                  <tr>
                    <th>Evaluation Metrics</th>
                    <th>Value</th>
                  </tr>
                </thead>
                <tbody>
                  ${this.results.map(
                    (row) => html`
                      <tr>
                        <td>${row["metric"] ?? ""}</td>
                        <td>${row["value"] ?? ""}</td>
                      </tr>
                    `
                  )}
                </tbody>
              </table>
            `
          : html`<p class="no-results">No results to display.</p>`}
      </div>
    `;
  }

  private async _handleButtonClick() {
    this._loading = true;
    this._error = "";

    try {
      const response = await fetch("http://localhost:8000/evaluate/");
      if (!response.ok) throw new Error(`Server error: ${response.status}`);

      const data = await response.json();
      this._evaluations = data.evaluations;

      if (this.selectedFile) {
        this._showFileResults(this.selectedFile);
      }
    } catch (err) {
      this._error = (err as Error).message;
    } finally {
      this._loading = false;
    }
  }

  connectedCallback() {
    super.connectedCallback();
    this._loadCachedEvaluations();
  }

  private async _loadCachedEvaluations() {
    try {
      const response = await fetch("http://localhost:8000/evaluations/");
      if (!response.ok) return;
      const data = await response.json();
      this._evaluations = data.evaluations;

      // If a file is already selected, show its results immediately
      if (this.selectedFile) {
        this._showFileResults(this.selectedFile);
      }
    } catch {
      // Silently fail — user can still run analysis manually
    }
  }
}