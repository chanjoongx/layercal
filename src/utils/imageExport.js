import html2canvas from 'html2canvas';

/**
 * Export DOM element to PNG image
 * @param {HTMLElement} element - Element to capture
 * @param {Object} options - Export options
 * @param {boolean} options.isDarkMode - Dark mode flag
 * @param {string} options.filename - Custom filename
 * @returns {Promise<boolean>} Success status
 */
export const exportToImage = async (element, options = {}) => {
  const { isDarkMode = false, filename, retryCount = 0 } = options;
  const MAX_RETRIES = 1;

  if (!element) {
    console.error('Export failed: No element provided');
    return false;
  }

  try {
    console.log(`Starting export (attempt ${retryCount + 1}/${MAX_RETRIES + 1})...`);
    
    if (document.fonts && document.fonts.ready) {
      await document.fonts.ready;
      console.log('Fonts loaded');
    }
    
    await new Promise(resolve => setTimeout(resolve, 100));

    const canvas = await html2canvas(element, {
      scale: 2,
      backgroundColor: isDarkMode ? '#1f2937' : '#ffffff',
      logging: false,
      useCORS: true,
      allowTaint: false,
      onclone: (clonedDoc) => {
        const ignoredElements = clonedDoc.querySelectorAll('[data-html2canvas-ignore]');
        ignoredElements.forEach(el => {
          el.style.display = 'none';
        });

        // Prevent text truncation
        const allElements = clonedDoc.body.querySelectorAll('*');
        allElements.forEach(el => {
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

        // Replace select elements with divs
        const selects = clonedDoc.querySelectorAll('select');
        const originalSelects = element.querySelectorAll('select');
        
        selects.forEach((select, idx) => {
          const originalSelect = originalSelects[idx];
          if (!originalSelect) return;

          const selectedText = select.options[select.selectedIndex]?.text || '';
          const div = clonedDoc.createElement('div');
          const computed = window.getComputedStyle(originalSelect);
          
          div.className = select.className;
          div.textContent = selectedText;
          div.style.cssText = `
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
          
          select.parentNode.replaceChild(div, select);
        });

        // Replace input elements with divs
        const inputs = clonedDoc.querySelectorAll('input[type="text"], input[type="number"]');
        const originalInputs = element.querySelectorAll('input[type="text"], input[type="number"]');
        
        inputs.forEach((input, idx) => {
          const originalInput = originalInputs[idx];
          if (!originalInput) return;

          const div = clonedDoc.createElement('div');
          const computed = window.getComputedStyle(originalInput);
          
          div.className = input.className;
          div.textContent = input.value || '';
          div.style.cssText = `
            font-family: ${computed.fontFamily};
            font-size: ${computed.fontSize};
            color: ${computed.color};
            background-color: ${computed.backgroundColor};
            border: ${computed.border};
            border-radius: ${computed.borderRadius};
            padding: ${computed.padding};
            width: ${computed.width};
            height: ${computed.height};
            text-align: ${computed.textAlign};
            display: flex;
            align-items: center;
            box-sizing: border-box;
            overflow: visible;
            white-space: normal;
          `;
          
          input.parentNode.replaceChild(div, input);
        });

        // Replace checkbox elements with styled divs
        const checkboxes = clonedDoc.querySelectorAll('input[type="checkbox"]');
        checkboxes.forEach(checkbox => {
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

        // Add watermark
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
        
        const captureArea = clonedDoc.body.querySelector('[data-capture-area]') || 
                           clonedDoc.body.firstChild;
        if (captureArea) {
          captureArea.style.position = 'relative';
          captureArea.appendChild(watermark);
        }
      }
    });

    console.log('Canvas created, converting to blob...');
    
    return new Promise((resolve) => {
      canvas.toBlob((blob) => {
        if (!blob) {
          console.error('Blob creation failed');
          resolve(false);
          return;
        }

        console.log(`Blob created: ${(blob.size / 1024 / 1024).toFixed(2)} MB`);
        
        try {
          const url = URL.createObjectURL(blob);
          const link = document.createElement('a');
          const timestamp = new Date().toISOString().slice(0, 10);
          
          link.download = filename || `layercal-model-${timestamp}.png`;
          link.href = url;
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
          
          console.log('Download complete');
          setTimeout(() => URL.revokeObjectURL(url), 100);
          resolve(true);
        } catch (err) {
          console.error('Download failed:', err);
          resolve(false);
        }
      }, 'image/png', 0.95);
    });

  } catch (error) {
    console.error('Export error:', error);
    
    if (retryCount < MAX_RETRIES) {
      console.log('Retrying...');
      await new Promise(resolve => setTimeout(resolve, 500));
      return exportToImage(element, { ...options, retryCount: retryCount + 1 });
    }
    
    return false;
  }
};

/**
 * Display export error message
 */
export const showExportError = (errorType = 'unknown') => {
  const messages = {
    'no-element': 'Export area not found. Please try again.',
    'timeout': 'Image generation timed out. Please try again.',
    'cors': 'Cannot export image due to CORS policy.',
    'unknown': 'Failed to export image. Please try again.'
  };

  const message = messages[errorType] || messages['unknown'];
  alert(message);
};

/**
 * Validate element before export
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