import { BrowserRouter, Route, Routes } from "react-router-dom";
import NavBar from "./components/NavBar";
import HomePage from "./pages/HomePage";
import ImageStitchingPage from "./pages/ImageStitchingPage";
import SiftComparePage from "./pages/SiftComparePage";
import "./App.css";

function App() {
  return (
    <BrowserRouter basename="/CV">
      <div className="app-shell">
        <NavBar />
        <main className="app-content">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/stitch" element={<ImageStitchingPage />} />
            <Route path="/sift" element={<SiftComparePage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;
