import { LitElement, html, css } from "lit";
import { customElement } from "lit/decorators.js";

@customElement("app-layout")
export class AppLayout extends LitElement {
  static styles = css`
    :host {
      display: flex;
      height: 100vh;
      width: 100vw;
      background-color: #f1f1f1;
      overflow: hidden;
    }

    .main-content {
      flex: 1;
      overflow-y: auto;
      padding: 20px;
      background-color: #f1f1f1;
      max-width: 100%;
      margin: 0 auto;
      height: 100vh;
      box-sizing: border-box;
    }

    .main-content::-webkit-scrollbar {
      width: 8px;
    }

    .main-content::-webkit-scrollbar-track {
      background: transparent;
    }

    .main-content::-webkit-scrollbar-thumb {
      background-color: #ccc;
      border-radius: 4px;
    }

    .main-content::-webkit-scrollbar-thumb:hover {
      background-color: #aaa;
    }

    .main-slot {
      display: block;
      max-width: 600px;
      margin: 0 auto;
    }
  `;

  render() {
    return html`
      <app-sidebar></app-sidebar>
      <div class="main-content">
        <slot class="main-slot"></slot>
      </div>
    `;
  }
}