import { useEffect, useState } from "react";
import Layout from "../../components/layout/Layout";
import Welcome from "../../components/home/Welcome";
import ChatBot from "../../components/home/ChatBot";

const LandingPage = () => {
  const [showWelcome, setShowWelcome] = useState(true);
  const [fadeOut, setFadeOut] = useState(false);

  useEffect(() => {
    const fadeTimer = setTimeout(() => {
      setFadeOut(true);
    }, 2500); // Start fading out after 2.5 seconds

    const removeTimer = setTimeout(() => {
      setShowWelcome(false);
    }, 3200); // Fully remove after fade (0.7s fade time)

    return () => {
      clearTimeout(fadeTimer);
      clearTimeout(removeTimer);
    };
  }, []);

  return (
    <Layout>
      <main className="font-sogeo flex flex-col gap-5 transition-all duration-700">
        {showWelcome && (
          <div
            className={`transition-all duration-700 ${
              fadeOut ? "opacity-0 -translate-y-5" : "opacity-100"
            }`}
          >
            <Welcome />
          </div>
        )}
        <ChatBot />
      </main>
    </Layout>
  );
};

export default LandingPage;
