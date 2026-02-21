import { useState } from "react";
import { Link, NavLink } from "react-router-dom";
import "./NavBar.css";

export function NavBar() {
  const [isOpen, setIsOpen] = useState(false);

  const closeMenu = () => setIsOpen(false);

  return (
    <header className="nav-root">
      <div className="nav-brand">
        <Link to="/" onClick={closeMenu}>
          World of Computer Vision
        </Link>
      </div>
      <button
        className="nav-toggle"
        type="button"
        aria-label="Toggle navigation"
        aria-expanded={isOpen}
        onClick={() => setIsOpen((prev) => !prev)}
      >
        Menu
      </button>
      <nav className={`nav-links${isOpen ? " open" : ""}`}>
        <NavLink to="/" end onClick={closeMenu}>
          Home
        </NavLink>
        <NavLink to="/measure" onClick={closeMenu}>
          Measure Lab
        </NavLink>
        <NavLink to="/stitch" onClick={closeMenu}>
          Image Stitching
        </NavLink>
        <NavLink to="/sift" onClick={closeMenu}>
          SIFT Compare
        </NavLink>
        <NavLink to="/module2" onClick={closeMenu}>
          Module 2
        </NavLink>
        <NavLink to="/module3" onClick={closeMenu}>
          Module 3
        </NavLink>
      </nav>
    </header>
  );
}

export default NavBar;
