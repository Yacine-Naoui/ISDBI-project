import React, { useEffect, useState } from "react";
import AiBrainAnimation from "./AiBrainAnimation";
import { motion } from "framer-motion";

const Hero = () => {
  const [showContent, setShowContent] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => setShowContent(true), 3000);
    return () => clearTimeout(timer);
  }, []);

  return (
    <section className="relative " id="hero">
      <motion.div
        initial={{ y: -2000, opacity: 0 }}
        animate={{ y: -1000, opacity: 1 }}
        transition={{ duration: 1.5, ease: "easeOut" }}
        className="absolute  top-32 left-1/2 -translate-x-1/2 w-[1400px] h-[1400px]    pointer-events-none"
      >
        <motion.div
          initial={{ opacity: 0, y: 0 }}
          animate={{ opacity: 0.5, y: 0 }}
          transition={{ duration: 1.8, ease: "easeOut" }}
          className="absolute bottom-0 left-1/2 -translate-x-1/2 w-[700px] h-[500px] bg-gradient-to-t from-cyan-400 to-transparent rounded-t-full blur-3xl pointer-events-none"
        />

        {/* Inner shadow (dark blue/black on top) */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 2, ease: "easeOut", delay: 0.3 }}
          className="absolute  top-32 left-1/2 -translate-x-1/2 w-[1400px] h-[1400px] pointer-events-none"
          style={{
            WebkitMaskImage: `
      radial-gradient(
        circle at center, 
        transparent 65%, 
        black 65%, 
        black 100%
      )
    `,
            WebkitMaskComposite: "exclude",
            maskImage: `
      radial-gradient(
        circle at center, 
        transparent 65%, 
        black 65%, 
        black 100%
      )
    `,
            maskComposite: "exclude",
            borderRadius: "9999px",
            boxShadow: `
              inset 0 80px 120px rgba(0, 30, 60, 0.95),
              inset 0 50px 80px rgba(0, 40, 80, 0.85),
              inset 0 30px 50px rgba(0, 60, 100, 0.7)
            `,
          }}
        />

        {/* Outer shadow (cyan/white on bottom) */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 2, ease: "easeOut", delay: 0.4 }}
          className="absolute  top-32 left-1/2 -translate-x-1/2 w-[1400px] h-[1400px] pointer-events-none"
          style={{
            WebkitMaskImage: `
      radial-gradient(
        circle at center, 
        transparent 65%, 
        black 65%, 
        black 100%
      )
    `,
            WebkitMaskComposite: "exclude",
            maskImage: `
      radial-gradient(
        circle at center, 
        transparent 65%, 
        black 65%, 
        black 100%
      )
    `,
            maskComposite: "exclude",
            borderRadius: "9999px",
            boxShadow: `
              0 -50px 150px rgba(0, 240, 255, 0.8),
              0 -30px 100px rgba(0, 220, 255, 0.9),
              0 -15px 60px rgba(255, 255, 255, 0.6),
              0 -8px 40px rgba(255, 255, 255, 0.8)
            `,
          }}
        />

        {/* Enhanced Arc with Animated Gradient - Transparent edges */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 2, ease: "easeOut", delay: 0.5 }}
          className="absolute  top-32 left-1/2 -translate-x-1/2 w-[1400px] h-[1400px] pointer-events-none"
          style={{
            WebkitMaskImage: `
      radial-gradient(
        circle at center, 
        transparent 65%, 
        black 65%, 
        black 100%
      )
    `,
            WebkitMaskComposite: "exclude",
            maskImage: `
      radial-gradient(
        circle at center, 
        transparent 65%, 
        black 65%, 
        black 100%
      )
    `,
            maskComposite: "exclude",
            borderRadius: "9999px",
            filter: "blur(1px)",
          }}
        >
          {/* Primary animated gradient - Transparent to light effect */}
          <motion.div
            className="w-full h-full"
            animate={{
              opacity: [0.85, 1, 0.9, 1, 0.85],
            }}
            transition={{
              duration: 4.5,
              repeat: Infinity,
              ease: "easeInOut",
            }}
            style={{
              borderRadius: "9999px",
              mixBlendMode: "screen",
              background:
                "linear-gradient(to top, rgba(255, 255, 255, 1) 0%, rgba(0, 228, 255, 0.95) 5%, rgba(0, 196, 255, 0.85) 15%, rgba(0, 153, 204, 0.6) 30%, rgba(0, 102, 153, 0.35) 45%, rgba(0, 61, 92, 0.15) 60%, transparent 75%, transparent 100%)",
            }}
          />
          
          {/* Secondary pulsing layer for enhanced light effect */}
          <motion.div
            className="absolute inset-0 w-full h-full"
            animate={{
              opacity: [0.4, 0.7, 0.5, 0.7, 0.4],
              scale: [1, 1.02, 1, 1.02, 1],
            }}
            transition={{
              duration: 5.5,
              repeat: Infinity,
              ease: "easeInOut",
              delay: 0.8,
            }}
            style={{
              borderRadius: "9999px",
              mixBlendMode: "screen",
              background:
                "radial-gradient(ellipse at bottom, rgba(255, 255, 255, 0.9) 0%, rgba(0, 240, 255, 0.7) 8%, rgba(0, 212, 255, 0.5) 18%, rgba(0, 196, 255, 0.3) 28%, transparent 50%)",
            }}
          />

          {/* Tertiary shimmer layer for extra depth */}
          <motion.div
            className="absolute inset-0 w-full h-full"
            animate={{
              opacity: [0.3, 0.6, 0.4, 0.6, 0.3],
            }}
            transition={{
              duration: 6.5,
              repeat: Infinity,
              ease: "easeInOut",
              delay: 1.2,
            }}
            style={{
              borderRadius: "9999px",
              mixBlendMode: "lighten",
              background:
                "linear-gradient(to top, rgba(0, 230, 255, 0.6) 0%, rgba(0, 196, 255, 0.4) 12%, rgba(0, 140, 200, 0.25) 25%, rgba(0, 100, 160, 0.12) 40%, transparent 55%)",
            }}
          />
        </motion.div>
      </motion.div>

      <div className="relative  min-h-screen flex flex-col justify-center items-center text-center px-6 pt-24 overflow-hidden">
        <div className="flex-1  ">
          {/* Title */}
          <motion.h1
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: showContent ? 1 : 0, y: showContent ? 0 : 40 }}
            transition={{ duration: 0.8, ease: "easeOut" }}
            className="text-[50px] md:text-[90px] font-bold leading-tight bg-gradient-to-t from-[#a4a4a4] to-white bg-clip-text text-transparent z-10"
          >
            Upgrade your world
          </motion.h1>

          {/* Subtitle */}
          <motion.p
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: showContent ? 1 : 0, y: showContent ? 0 : 40 }}
            transition={{ duration: 0.8, ease: "easeOut", delay: 0.3 }}
            className="text-[24px] md:text-[75px] font-bold leading-tight bg-gradient-to-t from-[#a4a4a4] to-white bg-clip-text text-transparent z-10"
          >
            with <span className="font-black  ">Sukai Labs</span>
          </motion.p>

          {/* Button */}
          {/* <motion.button
        initial={{ opacity: 0, y: 40 }}
        animate={{ opacity: showContent ? 1 : 0, y: showContent ? 0 : 40 }}
        transition={{ duration: 0.8, ease: "easeOut", delay: 0.6 }}
        className="mt-10 px-8 py-4 bg-white text-cyan-900 rounded-full text-lg font-semibold hover:bg-stone-100 transition z-10"
      >
        Discover Us
      </motion.button> */}
        </div>

        {/* Brain section comes BELOW text, not absolute */}
        <motion.div className="w-full flex-1       hidden md:block">
          {/* <AiBrainAnimation /> */}
        </motion.div>
      </div>
    </section>
  );
};

export default Hero;
