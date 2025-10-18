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

        {/* Enhanced Arc with Shadows and Animated Gradient */}
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
            filter: "blur(1.5px)",
            // Black shadow on inner (top) side, cyan shadow on outer (bottom) side
            boxShadow: `
              inset 0 60px 100px rgba(0, 0, 0, 0.9),
              inset 0 40px 70px rgba(0, 0, 0, 0.7),
              inset 0 20px 40px rgba(0, 0, 0, 0.5),
              0 -40px 120px rgba(0, 196, 255, 0.7),
              0 -20px 80px rgba(0, 196, 255, 0.9),
              0 -10px 50px rgba(0, 230, 255, 1)
            `,
          }}
        >
          {/* Primary animated gradient - Light growing effect */}
          <motion.div
            className="w-full h-full"
            animate={{
              opacity: [0.7, 1, 0.85, 1, 0.7],
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
                "linear-gradient(to top, #ffffff 0%, #ffffff 3%, #00e4ff 10%, #00c4ff 22%, #0099cc 38%, #006699 52%, #003d5c 65%, transparent 75%, transparent 100%)",
            }}
          />
          
          {/* Secondary pulsing layer for enhanced light effect */}
          <motion.div
            className="absolute inset-0 w-full h-full"
            animate={{
              opacity: [0.3, 0.6, 0.4, 0.65, 0.3],
              scale: [1, 1.015, 1, 1.015, 1],
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
                "radial-gradient(ellipse at bottom, #ffffff 0%, #00f0ff 8%, #00d4ff 18%, #00c4ff 30%, transparent 55%)",
            }}
          />

          {/* Tertiary shimmer layer for extra depth */}
          <motion.div
            className="absolute inset-0 w-full h-full"
            animate={{
              opacity: [0.2, 0.5, 0.3, 0.55, 0.2],
              rotate: [0, 5, 0, -5, 0],
            }}
            transition={{
              duration: 7,
              repeat: Infinity,
              ease: "easeInOut",
              delay: 1.5,
            }}
            style={{
              borderRadius: "9999px",
              mixBlendMode: "lighten",
              background:
                "linear-gradient(135deg, transparent 0%, #00c4ff 15%, #0099ff 25%, #0066cc 40%, transparent 60%)",
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
