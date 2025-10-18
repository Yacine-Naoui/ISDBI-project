import React, { useEffect, useState } from "react";

// Animated hero with masked arc: inner black shadow (top),
// outer cyan glow (bottom), and animated light gradient inside the arc.
const Hero = () => {
  const [showContent, setShowContent] = useState(false);

  useEffect(() => {
    const t = setTimeout(() => setShowContent(true), 1200);
    return () => clearTimeout(t);
  }, []);

  return (
    <section className="relative min-h-[80vh] md:min-h-screen flex items-center justify-center overflow-hidden" id="hero">
      {/* Big masked circle creating the glowing arc */}
      <div className="pointer-events-none absolute top-24 left-1/2 -translate-x-1/2 w-[1200px] h-[1200px] md:w-[1400px] md:h-[1400px] arc-slide">
        {/* Outer cyan glow on the bottom side (external side) */}
        <div
          className="absolute inset-0 blur-2xl"
          style={{
            WebkitMaskImage:
              "radial-gradient(circle at center, transparent var(--arc-cut,65%), black var(--arc-cut,65%), black 100%)",
            WebkitMaskComposite: "exclude",
            maskImage:
              "radial-gradient(circle at center, transparent var(--arc-cut,65%), black var(--arc-cut,65%), black 100%)",
            maskComposite: "exclude",
            borderRadius: "9999px",
            filter:
              "drop-shadow(0 60px 140px rgba(34,211,238,0.85)) drop-shadow(0 25px 50px rgba(34,211,238,0.65))",
          }}
        />

        {/* Animated inner light gradient: white -> cyan-400 -> deep blue, pulsing upward */}
        <div
          className="absolute inset-0"
          style={{
            WebkitMaskImage:
              "radial-gradient(circle at center, transparent var(--arc-cut,65%), black var(--arc-cut,65%), black 100%)",
            WebkitMaskComposite: "exclude",
            maskImage:
              "radial-gradient(circle at center, transparent var(--arc-cut,65%), black var(--arc-cut,65%), black 100%)",
            maskComposite: "exclude",
            borderRadius: "9999px",
            mixBlendMode: "screen",
            background:
              "linear-gradient(to top, rgba(255,255,255,0.95) 0%, rgba(34,211,238,0.95) 8%, rgba(14,165,233,0.9) 22%, rgba(14,165,233,0.2) 48%, rgba(14,165,233,0.0) 60%)",
            backgroundSize: "100% 200%",
            animation: "arcLightSweep 6s ease-in-out infinite",
            opacity: 0.95,
          }}
        />

        {/* Inner black shadow on the interior/top side */}
        <div
          className="absolute inset-0"
          style={{
            WebkitMaskImage:
              "radial-gradient(circle at center, transparent var(--arc-cut,65%), black var(--arc-cut,65%), black 100%)",
            WebkitMaskComposite: "exclude",
            maskImage:
              "radial-gradient(circle at center, transparent var(--arc-cut,65%), black var(--arc-cut,65%), black 100%)",
            maskComposite: "exclude",
            borderRadius: "9999px",
            background:
              "linear-gradient(to bottom, rgba(0,0,0,0.9) 4%, rgba(0,0,0,0.5) 18%, rgba(0,0,0,0.2) 34%, rgba(0,0,0,0.0) 55%)",
            opacity: 0.9,
          }}
        />

        {/* Subtle base haze below the arc for depth */}
        <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-[700px] h-[520px] rounded-t-full blur-3xl"
             style={{
               background:
                 "linear-gradient(to top, rgba(34,211,238,0.35), rgba(34,211,238,0.05) 60%, rgba(34,211,238,0) 100%)",
               opacity: 0.9,
             }}
        />
      </div>

      {/* Centered headline content (keeps simple; fades in) */}
      <div className={`relative z-10 text-center transition-all duration-700 ${showContent ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4"}`}>
        <h1 className="text-[40px] md:text-[78px] font-bold leading-tight bg-gradient-to-t from-[#9aa0a6] to-white bg-clip-text text-transparent">
          Boost ROI with AI Solutions
        </h1>
        <p className="mt-4 text-[18px] md:text-[24px] text-white/80 max-w-3xl mx-auto">
          Customizable AI that enhances user experience and streamlines operations.
        </p>
      </div>
    </section>
  );
};

export default Hero;
