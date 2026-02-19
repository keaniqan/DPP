import { LitElement, html, css } from "lit";
import { customElement, property, state } from "lit/decorators.js";


@customElement("sidebar-file")
export class SidebarFile extends LitElement {
  @property({ type: String })
  name = "";

  @property({ type: String })
  path = "";

  @state()
  private _showTooltip = false;

  @state()
  private _tooltipX = 0;

  @state()
  private _tooltipY = 0;

  static styles = css`
    .sidebar-file {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 0px 10px;
      margin: 0px 0px;
      border-radius: 4px;
      cursor: pointer;
      transition: background 0.2s;
      position: relative;
    }

    .sidebar-file span {
      color: #e8eaed;
      font-size: 10px;
    }
    .sidebar-file:hover {
      background: #2c5f7c;
    }

    .file-name {
      color: #e8eaed;
      font-size: 14px;
      flex: 1;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .tooltip {
      position: fixed;
      background: #1e1e1e;
      color: #e8eaed;
      font-size: 12px;
      padding: 4px 8px;
      border-radius: 4px;
      white-space: nowrap;
      z-index: 10000;
      pointer-events: none;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
      border: 1px solid #3c3c3c;
    }

    button {
      background: none;
      border: none;
      color: #e8eaed;
      cursor: pointer;
      padding: 4px;
      font-size: 16px;
    }

    button:hover {
      color: white;
    }
  `;

  private _onMouseEnter(e: MouseEvent) {
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    this._tooltipX = rect.left + 10;
    this._tooltipY = rect.bottom + 4;
    this._showTooltip = true;
  }

  private _onMouseLeave() {
    this._showTooltip = false;
  }

  private _onFileClick() {
    this.dispatchEvent(new CustomEvent("file-selected", {
      detail: { name: this.name, path: this.path },
      bubbles: true,
      composed: true,
    }));
  }

  render() {
    return html`
        <div class="sidebar-file"
             @click=${this._onFileClick}
             @mouseenter=${this._onMouseEnter}
             @mouseleave=${this._onMouseLeave}>
            <span class="file-name">${this.name}</span>
            ${this._showTooltip
              ? html`<span class="tooltip" style="left:${this._tooltipX}px;top:${this._tooltipY}px;">${this.name}</span>`
              : null}
            <popover-menu position="bottom">
                <button slot="trigger">⋮</button>
            </popover-menu>
        </div>
    `;
  }
}
