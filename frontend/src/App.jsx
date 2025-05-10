import { createRoot } from "react-dom/client";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import "./index.css";
import LandingPage from "./pages/home/LandingPage";
import ReportPage from "./pages/report/ReportPage";
import StandardsPage from "./pages/standards/StandardsPage";
import ReportPage2 from "./pages/report_agent2/ReportPage2";
import ReportPage3 from "./pages/report_agent3/ReportPage3";
import ReportPage4 from "./pages/report_agent4/ReportPage4";

function App() {
  return (
    <>
      <BrowserRouter>
        <Routes>
          {/* Route for the prompt or chat page */}
          <Route exact path={"/"} element={<LandingPage />} />

          {/* Route for the report page, with dynamic reportId */}
          <Route exact path={"/report/:reportId"} element={<ReportPage />} />
          <Route exact path={"/report"} element={<ReportPage />} />

          <Route exact path={"/report2/:report2Id"} element={<ReportPage2 />} />
          <Route exact path={"/report2"} element={<ReportPage2 />} />

          <Route exact path={"/report3/:report3Id"} element={<ReportPage3 />} />
          <Route exact path={"/report3"} element={<ReportPage3 />} />

          <Route exact path={"/report4/:report4Id"} element={<ReportPage4 />} />
          <Route exact path={"/report4"} element={<ReportPage4 />} />
          {/* Route for standards */}
          <Route exact path={"/stand"} element={<StandardsPage />} />
        </Routes>
      </BrowserRouter>
    </>
  );
}

export default App;

const root = createRoot(document.getElementById("root"));
root.render(<App />);
