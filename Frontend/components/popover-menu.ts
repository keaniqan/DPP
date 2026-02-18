import { LitElement, html, css } from "lit";
import { customElement, property, state } from "lit/decorators.js";

@customElement("popover-menu")
export class PopoverMenu extends LitElement {
  @property({ type: Boolean }) open = false;
  @property({ type: String }) position = "bottom"; // top, bottom, left, right

  static styles = css`
    :host {
      position: relative;
      display: inline-block;
    }

    .trigger {
      cursor: pointer;
    }

    .popover-container {
      position: absolute;
      z-index: 1000;
      opacity: 0;
      visibility: hidden;
      transition: opacity 0.2s ease, visibility 0.2s ease;
    }

    .popover-container.open {
      opacity: 1;
      visibility: visible;
    }

    .popover-container.bottom {
      top: calc(100% + 8px);
      left: 50%;
      transform: translateX(-50%);
    }

    .popover-container.top {
      bottom: calc(100% + 8px);
      left: 50%;
      transform: translateX(-50%);
    }

    .popover-container.left {
      right: calc(100% + 8px);
      top: 50%;
      transform: translateY(-50%);
    }

    .popover-container.right {
      left: calc(100% + 8px);
      top: 50%;
      transform: translateY(-50%);
    }

    .popover-content {
      background: #1e3a52;
      border: 1px solid #2c5f7c;
      border-radius: 8px;
      padding: 8px;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
      min-width: 200px;
    }

    .arrow {
      position: absolute;
      width: 0;
      height: 0;
      border-style: solid;
    }

    .popover-container.bottom .arrow {
      top: -8px;
      left: 50%;
      transform: translateX(-50%);
      border-width: 0 8px 8px 8px;
      border-color: transparent transparent #1e3a52 transparent;
    }

    .popover-container.top .arrow {
      bottom: -8px;
      left: 50%;
      transform: translateX(-50%);
      border-width: 8px 8px 0 8px;
      border-color: #1e3a52 transparent transparent transparent;
    }

    .menu-item {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 12px 16px;
      background: #2c5f7c;
      border: 1px solid #3d7a9a;
      border-radius: 6px;
      color: #e8eaed;
      cursor: pointer;
      transition: background 0.2s ease;
      font-size: 14px;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    .menu-item:hover {
      background: #3d7a9a;
    }

    .menu-item svg {
      width: 20px;
      height: 20px;
      fill: currentColor;
    }

    .backdrop {
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      z-index: 999;
      display: none;
    }

    .backdrop.open {
      display: block;
    }
  `;

  private handleToggle() {
    this.open = !this.open;
  }

  private handleClose() {
    this.open = false;
  }

  private handleDelete() {
    this.dispatchEvent(new CustomEvent("delete", { bubbles: true, composed: true }));
    this.handleClose();
  }

  render() {
    return html`
      <div class="backdrop ${this.open ? "open" : ""}" @click="${this.handleClose}"></div>
      
      <div class="trigger" @click="${this.handleToggle}">
        <slot name="trigger"></slot>
      </div>

      <div class="popover-container ${this.open ? "open" : ""} ${this.position}">
        <div class="arrow"></div>
        <div class="popover-content">
          <div class="menu-item" @click="${this.handleDelete}">
            <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/>
            </svg>
            <span>Delete</span>
          </div>
          <slot name="menu-items"></slot>
        </div>
      </div>
    `;
  }
}
