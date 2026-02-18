import { html, css, LitElement } from "lit";
import { customElement, property } from "lit/decorators.js";

@customElement("base-component")
export class BaseComponent extends LitElement {
  static styles = css`
    :host {
      display: block;
    }
    `;
render() {
    return html`<slot></slot>`;
  }
}