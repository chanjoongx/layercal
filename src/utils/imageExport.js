/**
 * PNG export of the model canvas.
 *
 * html2canvas is ~200 KB and only ever needed once the user clicks Export, so
 * it is dynamically imported instead of being bundled into the initial load.
 */

const MAX_RETRIES = 1;

/**
 * Export a DOM element to a PNG download.
 *
 * @param {HTMLElement} element - Element to capture
 * @param {Object} options
 * @param {boolean} options.isDarkMode - Dark mode flag
 * @param {string} [options.filename] - Custom filename
 * @returns {Promise<{ ok: boolean, error?: string }>} Structured result; `error` is an i18n key
 */
export const exportToImage = async (element, options = {}) => {
  const { isDarkMode = false, filename, retryCount = 0 } = options;

  if (!element) {
    return { ok: false, error: 'no-element' };
  }

  let html2canvas;
  try {
    ({ default: html2canvas } = await import('html2canvas'));
  } catch {
    return { ok: false, error: 'load-failed' };
  }

  try {
    // Web fonts that haven't loaded render as fallbacks in the capture.
    if (document.fonts?.ready) {
      await document.fonts.ready;
    }
    await new Promise(resolve => setTimeout(resolve, 100));

    const canvas = await html2canvas(element, {
      scale: 2,
      // Matches --background in index.css, so the export does not sit on a
      // different ground colour from the app it is a picture of.
      backgroundColor: isDarkMode ? '#0b0d13' : '#fcfcfd',
      logging: false,
      useCORS: true,
      allowTaint: false,
      onclone: (clonedDoc, clonedElement) => {
        // Scope every lookup to the cloned capture area. Querying the whole
        // cloned document would misalign the index-based pairing below as soon
        // as a form control exists anywhere else on the page.
        const root = clonedElement
          || clonedDoc.querySelector('[data-capture-area]')
          || clonedDoc.body;

        // Anything visually hidden must stay hidden in a screenshot. `.sr-only`
        // hides itself with a 1px box plus `overflow: hidden`, which the
        // truncation fix below would undo — turning every screen-reader label
        // and the viewer's hidden layer list into visible text down the margin.
        root.querySelectorAll('[data-html2canvas-ignore], .sr-only').forEach(el => {
          el.style.display = 'none';
        });

        // Prevent text truncation
        root.querySelectorAll('*').forEach(el => {
          const computed = window.getComputedStyle(el);

          if (computed.overflow === 'hidden') {
            el.style.overflow = 'visible';
          }
          if (computed.textOverflow === 'ellipsis') {
            el.style.textOverflow = 'clip';
          }
          if (computed.whiteSpace === 'nowrap') {
            el.style.whiteSpace = 'normal';
          }
        });

        // Form controls don't render their value in a canvas capture, so swap
        // each one for a div carrying the same text and computed styling.
        const copyBoxStyles = (computed) => `
          font-family: ${computed.fontFamily};
          font-size: ${computed.fontSize};
          color: ${computed.color};
          background-color: ${computed.backgroundColor};
          border: ${computed.border};
          border-radius: ${computed.borderRadius};
          padding: ${computed.padding};
          width: ${computed.width};
          height: ${computed.height};
          display: flex;
          align-items: center;
          box-sizing: border-box;
          overflow: visible;
          white-space: normal;
        `;

        const selects = root.querySelectorAll('select');
        const originalSelects = element.querySelectorAll('select');

        selects.forEach((select, idx) => {
          const originalSelect = originalSelects[idx];
          if (!originalSelect) return;

          const selectedText = select.options[select.selectedIndex]?.text || '';
          const div = clonedDoc.createElement('div');

          div.className = select.className;
          div.textContent = selectedText;
          div.style.cssText = copyBoxStyles(window.getComputedStyle(originalSelect));

          select.parentNode.replaceChild(div, select);
        });

        const inputs = root.querySelectorAll('input[type="text"], input[type="number"]');
        const originalInputs = element.querySelectorAll('input[type="text"], input[type="number"]');

        inputs.forEach((input, idx) => {
          const originalInput = originalInputs[idx];
          if (!originalInput) return;

          const computed = window.getComputedStyle(originalInput);
          const div = clonedDoc.createElement('div');

          div.className = input.className;
          div.textContent = originalInput.value || '';
          div.style.cssText = copyBoxStyles(computed) + `text-align: ${computed.textAlign};`;

          input.parentNode.replaceChild(div, input);
        });

        root.querySelectorAll('input[type="checkbox"]').forEach(checkbox => {
          const div = clonedDoc.createElement('div');
          div.style.cssText = `
            width: 16px;
            height: 16px;
            border: 2px solid ${checkbox.checked ? '#8b5cf6' : '#d1d5db'};
            background-color: ${checkbox.checked ? '#8b5cf6' : 'transparent'};
            border-radius: 4px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
          `;

          if (checkbox.checked) {
            div.innerHTML = `<svg viewBox="0 0 16 16" style="width:100%;height:100%;fill:white;">
              <path d="M13.854 3.646a.5.5 0 0 1 0 .708l-7 7a.5.5 0 0 1-.708 0l-3.5-3.5a.5.5 0 1 1 .708-.708L6.5 10.293l6.646-6.647a.5.5 0 0 1 .708 0z"/>
            </svg>`;
          }

          checkbox.parentNode.replaceChild(div, checkbox);
        });

        // Watermark
        const watermark = clonedDoc.createElement('div');
        watermark.style.cssText = `
          position: absolute;
          bottom: 16px;
          right: 16px;
          background: ${isDarkMode ? 'rgba(0,0,0,0.7)' : 'rgba(255,255,255,0.9)'};
          padding: 8px 16px;
          border-radius: 8px;
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
          font-size: 14px;
          font-weight: 600;
          color: ${isDarkMode ? '#a78bfa' : '#7c3aed'};
          box-shadow: 0 2px 8px rgba(0,0,0,0.1);
          z-index: 9999;
        `;
        watermark.textContent = 'LayerCal • layercal.com';

        root.style.position = 'relative';
        root.appendChild(watermark);
      }
    });

    const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'));
    if (!blob) {
      return { ok: false, error: 'unknown' };
    }

    const url = URL.createObjectURL(blob);
    try {
      const link = document.createElement('a');
      const timestamp = new Date().toISOString().slice(0, 10);

      link.download = filename || `layercal-model-${timestamp}.png`;
      link.href = url;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      return { ok: true };
    } finally {
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    }

  } catch (error) {
    if (retryCount < MAX_RETRIES) {
      await new Promise(resolve => setTimeout(resolve, 500));
      return exportToImage(element, { ...options, retryCount: retryCount + 1 });
    }

    console.error('Image export failed:', error);
    const isCORS = /tainted|cross-origin|SecurityError/i.test(String(error?.message || error));
    return { ok: false, error: isCORS ? 'cors' : 'unknown' };
  }
};

/**
 * Validate element before export.
 * Error codes: 'no-element' | 'empty-element'. The caller localises them,
 * alongside 'load-failed' | 'cors' | 'unknown' from exportToImage.
 */
export const validateExportElement = (element) => {
  if (!element) {
    return { valid: false, error: 'no-element' };
  }

  const rect = element.getBoundingClientRect();
  if (rect.width === 0 || rect.height === 0) {
    return { valid: false, error: 'empty-element' };
  }

  return { valid: true, error: null };
};
