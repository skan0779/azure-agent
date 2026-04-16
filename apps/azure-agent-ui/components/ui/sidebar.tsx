"use client";

import * as React from "react";
import { PanelLeftIcon } from "lucide-react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

type SidebarContextValue = {
  isMobile: boolean;
  open: boolean;
  setOpen: React.Dispatch<React.SetStateAction<boolean>>;
  openMobile: boolean;
  setOpenMobile: React.Dispatch<React.SetStateAction<boolean>>;
  toggleSidebar: () => void;
};

const SidebarContext = React.createContext<SidebarContextValue | null>(null);

export const useSidebar = () => {
  const context = React.useContext(SidebarContext);

  if (!context) {
    throw new Error("useSidebar must be used within a SidebarProvider.");
  }

  return context;
};

export const SidebarProvider = ({
  defaultOpen = true,
  children,
}: {
  defaultOpen?: boolean;
  children: React.ReactNode;
}) => {
  const [open, setOpen] = React.useState(defaultOpen);
  const [openMobile, setOpenMobile] = React.useState(false);
  const [isMobile, setIsMobile] = React.useState(false);

  React.useEffect(() => {
    const mediaQuery = window.matchMedia("(max-width: 767px)");

    const handleChange = () => {
      setIsMobile(mediaQuery.matches);
    };

    handleChange();
    mediaQuery.addEventListener("change", handleChange);

    return () => {
      mediaQuery.removeEventListener("change", handleChange);
    };
  }, []);

  const toggleSidebar = React.useCallback(() => {
    if (isMobile) {
      setOpenMobile((value) => !value);
      return;
    }

    setOpen((value) => !value);
  }, [isMobile]);

  const value = React.useMemo(
    () => ({
      isMobile,
      open,
      setOpen,
      openMobile,
      setOpenMobile,
      toggleSidebar,
    }),
    [isMobile, open, openMobile, toggleSidebar],
  );

  return (
    <SidebarContext.Provider value={value}>
      <div
        data-slot="sidebar-wrapper"
        className="flex h-dvh w-full overflow-hidden"
      >
        {children}
      </div>
    </SidebarContext.Provider>
  );
};

export const Sidebar = ({
  className,
  children,
}: {
  className?: string;
  children: React.ReactNode;
}) => {
  const { isMobile, open, openMobile, setOpenMobile } = useSidebar();

  if (isMobile) {
    return (
      <>
        <div
          data-slot="sidebar-overlay"
          data-state={openMobile ? "open" : "closed"}
          className={cn(
            "fixed inset-0 z-40 bg-black/40 transition-opacity md:hidden",
            openMobile ? "opacity-100" : "pointer-events-none opacity-0",
          )}
          onClick={() => setOpenMobile(false)}
        />
        <aside
          data-slot="sidebar"
          data-mobile="true"
          data-state={openMobile ? "open" : "closed"}
          className={cn(
            "fixed inset-y-0 left-0 z-50 flex w-72 flex-col border-r border-white/10 bg-[#171717]/95 backdrop-blur-md transition-transform duration-200 md:hidden",
            openMobile ? "translate-x-0" : "-translate-x-full",
            className,
          )}
        >
          {children}
        </aside>
      </>
    );
  }

  return (
    <aside
      data-slot="sidebar"
      data-state={open ? "open" : "collapsed"}
      className={cn(
        "hidden h-dvh shrink-0 overflow-hidden bg-[#171717]/95 backdrop-blur-md transition-[width] duration-200 md:flex",
        open ? "w-72 border-r border-white/10" : "w-0 border-r-0",
        className,
      )}
    >
      <div className="flex h-full w-72 flex-col">{children}</div>
    </aside>
  );
};

export const SidebarHeader = ({
  className,
  ...props
}: React.ComponentProps<"div">) => {
  return (
    <div
      data-slot="sidebar-header"
      className={cn("flex flex-col", className)}
      {...props}
    />
  );
};

export const SidebarContent = ({
  className,
  ...props
}: React.ComponentProps<"div">) => {
  return (
    <div
      data-slot="sidebar-content"
      className={cn("min-h-0 flex-1", className)}
      {...props}
    />
  );
};

export const SidebarInset = ({
  className,
  ...props
}: React.ComponentProps<"div">) => {
  return (
    <div
      data-slot="sidebar-inset"
      className={cn("relative flex min-h-0 min-w-0 flex-1 flex-col", className)}
      {...props}
    />
  );
};

export const SidebarTrigger = ({
  className,
  onClick,
  ...props
}: React.ComponentProps<typeof Button>) => {
  const { isMobile, open, openMobile, toggleSidebar } = useSidebar();
  const isOpen = isMobile ? openMobile : open;

  return (
    <Button
      data-slot="sidebar-trigger"
      variant="ghost"
      size="icon"
      aria-label="Toggle sidebar"
      aria-pressed={isOpen}
      className={cn("size-8 rounded-full", className)}
      onClick={(event) => {
        onClick?.(event);
        if (event.defaultPrevented) {
          return;
        }

        toggleSidebar();
      }}
      {...props}
    >
      <PanelLeftIcon className="size-4" />
    </Button>
  );
};
