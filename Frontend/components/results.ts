import { html, css, LitElement } from "lit";
import { customElement, property } from "lit/decorators.js";

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
  `;

  @property({ type: Array }) results: Array<Record<string, string>> = [];

  render() {
    return html`
      <div class="results-container">
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
}