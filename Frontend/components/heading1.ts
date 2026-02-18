import { html, css, LitElement } from "lit";
import { customElement, property } from "lit/decorators.js";

@customElement("heading-1")
export class Heading1 extends LitElement {
  static styles = css`
    :host {
      display: block;
    }
    .heading-text {
      font-size: 16px;
      font-weight: bold;
      color: #000000;
      margin-top: 0px;
      margin-bottom: 10px;
    }
    `;

    @property({ type: String })
    text = "";
    
  render() {
      return html`
          <p class="heading-text">
              ${this.text}
          </p>`;
    }
}