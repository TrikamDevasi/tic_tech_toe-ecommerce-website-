import { Menu, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { NAV_ITEMS } from "@/landing/claims";

function focusableElements(container: HTMLElement) {
  return Array.from(container.querySelectorAll<HTMLElement>(
    'a[href], button:not([disabled]), [tabindex]:not([tabindex="-1"])',
  ));
}

export default function LandingNavbar() {
  const [isOpen, setIsOpen] = useState(false);
  const [isCompact, setIsCompact] = useState(false);
  const drawerRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    const update = () => setIsCompact(window.scrollY > 18);
    update();
    window.addEventListener("scroll", update, { passive: true });
    return () => window.removeEventListener("scroll", update);
  }, []);

  useEffect(() => {
    if (!isOpen) return;
    const trigger = triggerRef.current;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setIsOpen(false);
      if (event.key !== "Tab" || !drawerRef.current) return;
      const items = focusableElements(drawerRef.current);
      if (!items.length) return;
      const first = items[0];
      const last = items[items.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", closeOnEscape);
    window.setTimeout(() => focusableElements(drawerRef.current ?? document.body)[0]?.focus(), 0);
    return () => {
      document.body.style.overflow = previousOverflow;
      document.removeEventListener("keydown", closeOnEscape);
      trigger?.focus();
    };
  }, [isOpen]);

  const closeDrawer = () => setIsOpen(false);

  return (
    <header className={`landing-nav ${isCompact ? "is-compact" : ""}`}>
      <div className="landing-shell landing-nav__inner">
        <a className="landing-brand" href="#top" aria-label="PriceIQ home">
          <span className="landing-brand__mark" aria-hidden="true"><i /><i /><i /></span>
          <span>PriceIQ</span>
        </a>

        <nav className="landing-nav__links" aria-label="Primary navigation">
          {NAV_ITEMS.map(([label, href]) => <a key={href} href={href}>{label}</a>)}
        </nav>

        <Link className="landing-nav__cta" to="/shop">Open platform <span aria-hidden="true">↗</span></Link>
        <button
          ref={triggerRef}
          className="landing-menu-button"
          type="button"
          onClick={() => setIsOpen(true)}
          aria-label="Open navigation menu"
          aria-expanded={isOpen}
          aria-controls="landing-mobile-menu"
        >
          <Menu size={20} aria-hidden="true" />
        </button>
      </div>

      {isOpen && (
        <div className="landing-drawer-backdrop" onMouseDown={closeDrawer}>
          <div
            ref={drawerRef}
            className="landing-drawer"
            id="landing-mobile-menu"
            role="dialog"
            aria-modal="true"
            aria-label="Navigation menu"
            onMouseDown={(event) => event.stopPropagation()}
          >
            <div className="landing-drawer__top">
              <span className="landing-brand"><span className="landing-brand__mark" aria-hidden="true"><i /><i /><i /></span>PriceIQ</span>
              <button className="landing-menu-button" type="button" onClick={closeDrawer} aria-label="Close navigation menu"><X size={20} /></button>
            </div>
            <nav className="landing-drawer__links" aria-label="Mobile navigation">
              {NAV_ITEMS.map(([label, href]) => <a key={href} href={href} onClick={closeDrawer}>{label}</a>)}
              <Link to="/shop" onClick={closeDrawer}>Open platform <span aria-hidden="true">↗</span></Link>
            </nav>
            <p className="landing-drawer__note">Press Escape to close</p>
          </div>
        </div>
      )}
    </header>
  );
}
