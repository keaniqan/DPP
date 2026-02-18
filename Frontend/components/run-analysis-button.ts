import {html, css, LitElement} from 'lit';
import {customElement, property} from 'lit/decorators.js';

@customElement('run-analysis-button')
export class RunAnalysisButton extends LitElement {
  static styles = css`
    :host {
      display: inline-block;
    }

    .run-button {
      display: flex;
      align-items: center;
      gap: 16px;
      background: #4a4a4ac7;
      border: none;
      border-radius: 20px;
      padding: 16px 40px 16px 20px;
      cursor: pointer;
      transition: all 0.3s ease;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }

    .run-button:hover {
      background: #5a5a5a;
      transform: translateY(-2px);
      box-shadow: 0 6px 16px rgba(0, 0, 0, 0.4);
    }

    .run-button:active {
      transform: translateY(0);
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
    }

    .play-icon {
      width: 24px;
      height: 24px;
      background: white;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
    }

    .play-icon svg {
      width: 24px;
      height: 24px;
      fill: #4a4a4a;
      margin-left: 4px;
    }

    .button-text {
      color: white;
      font-size: 18px;
      font-weight: bold;
      text-transform: uppercase;
      letter-spacing: 1px;
      font-family: Arial, sans-serif;
    }

    .run-button:disabled {
      opacity: 0.6;
      cursor: not-allowed;
    }

    .run-button:disabled:hover {
      transform: none;
      background: #4a4a4a;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
  `;

  @property({type: Boolean})
  disabled = false;

  private handleClick() {
    if (!this.disabled) {
      this.dispatchEvent(new CustomEvent('run-analysis', {
        bubbles: true,
        composed: true
      }));
    }
  }

  render() {
    return html`
      <!-- <button 
        class="run-button" 
        ?disabled=${this.disabled}
        @click=${this.handleClick}
      >
      </button> -->
    `;
  }
}
