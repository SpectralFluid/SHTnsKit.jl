// SHTnsKit.jl documentation UX enhancements

document.addEventListener("DOMContentLoaded", function () {
    // --- Back-to-top button ---
    var btn = document.createElement("button");
    btn.className = "back-to-top";
    btn.setAttribute("aria-label", "Back to top");
    btn.innerHTML = "↑";
    document.body.appendChild(btn);

    window.addEventListener("scroll", function () {
        if (window.scrollY > 400) {
            btn.classList.add("visible");
        } else {
            btn.classList.remove("visible");
        }
    }, { passive: true });

    btn.addEventListener("click", function () {
        window.scrollTo({ top: 0, behavior: "smooth" });
    });

    // --- Figure lightbox ---
    var figures = document.querySelectorAll(".grid-pattern-figure");
    figures.forEach(function (fig) {
        fig.addEventListener("click", function () {
            var img = fig.querySelector("img");
            if (!img) return;

            var overlay = document.createElement("div");
            overlay.className = "figure-lightbox";

            var clone = document.createElement("img");
            clone.src = img.src;
            clone.alt = img.alt;
            overlay.appendChild(clone);
            document.body.appendChild(overlay);

            requestAnimationFrame(function () {
                overlay.classList.add("active");
            });

            function close() {
                overlay.classList.remove("active");
                setTimeout(function () { overlay.remove(); }, 250);
            }

            overlay.addEventListener("click", close);
            document.addEventListener("keydown", function handler(e) {
                if (e.key === "Escape") {
                    close();
                    document.removeEventListener("keydown", handler);
                }
            });
        });
    });
});
