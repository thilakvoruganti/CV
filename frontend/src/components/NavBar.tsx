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
        <NavLink to="/stitch">Image Stitching</NavLink>
        <NavLink to="/sift">SIFT Compare</NavLink>
      </nav>
    </header>
  );
}

export default NavBar;
