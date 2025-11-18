import { Link, NavLink } from "react-router-dom";
import "./NavBar.css";

export function NavBar() {
  return (
    <header className="nav-root">
      <div className="nav-brand">
        <Link to="/">World of Computer Vision</Link>
      </div>
      <nav className="nav-links">
        <NavLink to="/" end>
          Home
        </NavLink>
        <NavLink to="/measure">Measure Lab</NavLink>
        <NavLink to="/stitch">Image Stitching</NavLink>
        <NavLink to="/sift">SIFT Compare</NavLink>
        <NavLink to="/module2">Module 2</NavLink>
        <NavLink to="/module3">Module 3</NavLink>
      </nav>
    </header>
  );
}

export default NavBar;
