import React, { useCallback, useEffect, useRef } from 'react';

const FOCUSABLE = [
  'a[href]',
  'button:not([disabled])',
  'textarea:not([disabled])',
  'input:not([disabled]):not([type="hidden"])',
  'select:not([disabled])',
  '[tabindex]:not([tabindex="-1"])',
].join(',');

/**
 * Accessible dialog shell.
 *
 * Handles the four things every modal needs and that are easy to forget:
 * focus moves in on open, Tab cannot escape while open, Escape closes, and
 * focus returns to whatever opened it. Background scrolling is locked too.
 *
 * @param {string} labelledBy  id of the element naming the dialog
 * @param {() => void} onClose called for Escape and backdrop clicks
 * @param {() => boolean} [onEscape] return true to swallow Escape (for a
 *        nested popover that should close first)
 */
export default function Modal({
  isDarkMode,
  labelledBy,
  onClose,
  onEscape,
  className = '',
  initialFocusRef,
  children,
}) {
  const dialogRef = useRef(null);
  const backdropRef = useRef(null);

  // Handlers live in refs so the setup effect runs exactly once. If it re-ran
  // whenever a caller passed a new closure, its cleanup would restore focus to
  // the opener mid-session and yank focus out of the dialog.
  const onCloseRef = useRef(onClose);
  const onEscapeRef = useRef(onEscape);
  onCloseRef.current = onClose;
  onEscapeRef.current = onEscape;

  useEffect(() => {
    const dialog = dialogRef.current;
    const previouslyFocused = document.activeElement;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';

    // Move focus into the dialog, otherwise the next Tab lands behind it.
    const target = initialFocusRef?.current
      || dialog?.querySelector(FOCUSABLE)
      || dialog;
    target?.focus?.({ preventScroll: true });

    const handleKeyDown = (e) => {
      if (e.key === 'Escape') {
        if (onEscapeRef.current && onEscapeRef.current() === true) return;
        e.stopPropagation();
        onCloseRef.current();
        return;
      }

      if (e.key !== 'Tab' || !dialog) return;

      const items = Array.from(dialog.querySelectorAll(FOCUSABLE))
        .filter(el => el.offsetParent !== null || el === document.activeElement);
      if (items.length === 0) {
        e.preventDefault();
        return;
      }

      const first = items[0];
      const last = items[items.length - 1];

      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      } else if (!dialog.contains(document.activeElement)) {
        e.preventDefault();
        first.focus();
      }
    };

    document.addEventListener('keydown', handleKeyDown, true);
    return () => {
      document.removeEventListener('keydown', handleKeyDown, true);
      document.body.style.overflow = previousOverflow;
      previouslyFocused?.focus?.({ preventScroll: true });
    };
    // Runs once per mount: the dialog is mounted only while it is open.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Closing on mousedown rather than click means selecting text inside the
  // dialog and releasing over the backdrop no longer dismisses it.
  const handleBackdropMouseDown = useCallback((e) => {
    if (e.target === backdropRef.current) onClose();
  }, [onClose]);

  return (
    <div
      ref={backdropRef}
      onMouseDown={handleBackdropMouseDown}
      className={`fixed inset-0 z-50 flex items-center justify-center p-4 ${
        isDarkMode ? 'bg-black/70' : 'bg-black/50'
      }`}
    >
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={labelledBy}
        tabIndex={-1}
        className={`relative rounded-2xl shadow-2xl transition-all duration-200 ${
          isDarkMode ? 'bg-gray-800' : 'bg-white'
        } ${className}`}
      >
        {children}
      </div>
    </div>
  );
}
