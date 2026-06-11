import { useEffect, useState } from "react";
import { motion } from "motion/react";
import { ease } from "@konjoai/ui";

export interface NavItem {
  id: string;
  label: string;
}

export interface SectionNavProps {
  items: NavItem[];
}

/**
 * Sticky right-rail section navigator with scroll-spy. Highlights the section
 * currently in view via IntersectionObserver and smooth-scrolls on click — the
 * spine that turns a long cockpit into something you can fly through.
 */
export function SectionNav({ items }: SectionNavProps) {
  const [active, setActive] = useState<string>(items[0]?.id ?? "");

  useEffect(() => {
    if (typeof IntersectionObserver !== "function") return;
    const obs = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
        if (visible) setActive(visible.target.id);
      },
      { rootMargin: "-30% 0px -60% 0px", threshold: [0, 0.25, 0.5, 1] },
    );
    for (const it of items) {
      const el = document.getElementById(it.id);
      if (el) obs.observe(el);
    }
    return () => obs.disconnect();
  }, [items]);

  const go = (id: string) => {
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  return (
    <nav
      aria-label="sections"
      className="hidden xl:flex flex-col gap-1 fixed right-6 top-1/2 -translate-y-1/2 z-20"
    >
      {items.map((it) => {
        const on = it.id === active;
        return (
          <button
            key={it.id}
            type="button"
            onClick={() => go(it.id)}
            aria-current={on ? "true" : undefined}
            className="group flex items-center gap-2 justify-end"
            title={it.label}
          >
            <span
              className={[
                "text-konjo-mono uppercase tracking-[0.18em] text-[9px] transition-opacity",
                on ? "opacity-100 text-konjo-accent" : "opacity-0 group-hover:opacity-100 text-konjo-fg-muted",
              ].join(" ")}
            >
              {it.label}
            </span>
            <motion.span
              layout
              className="rounded-full"
              animate={{
                width: on ? 22 : 10,
                backgroundColor: on ? "var(--color-konjo-accent)" : "var(--color-konjo-line)",
              }}
              transition={{ duration: 0.3, ease: ease.kanjo }}
              style={{ height: 3, boxShadow: on ? "0 0 8px var(--color-konjo-accent)" : "none" }}
            />
          </button>
        );
      })}
    </nav>
  );
}
