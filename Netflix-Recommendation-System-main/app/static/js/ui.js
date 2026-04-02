(function () {
  function setStaggerFallbacks() {
    var revealItems = document.querySelectorAll(".reveal");
    revealItems.forEach(function (item, index) {
      if (!item.style.getPropertyValue("--stagger")) {
        item.style.setProperty("--stagger", String(index));
      }
    });
  }

  function countSelections(selectElement) {
    if (!selectElement) {
      return 0;
    }

    var selected = Array.from(selectElement.options).filter(function (option) {
      return option.selected;
    });
    return selected.length;
  }

  function updateCounter(selectElement, targetId) {
    var counter = document.getElementById(targetId);
    if (!counter || !selectElement) {
      return;
    }

    counter.textContent = String(countSelections(selectElement));
  }

  function initializeMultiselect(selectElement, placeholder) {
    if (!window.jQuery || !window.jQuery.fn || typeof window.jQuery.fn.multiselect !== "function") {
      return;
    }

    if (!selectElement || selectElement.dataset.multiselectReady === "true") {
      return;
    }

    window.jQuery(selectElement).multiselect({
      columns: 1,
      placeholder: placeholder,
      search: true,
      searchOptions: {
        default: "Contains"
      },
      selectAll: true
    });

    selectElement.dataset.multiselectReady = "true";
  }

  function initializePreferenceForm() {
    var form = document.getElementById("preference-form");
    var titlesSelect = document.getElementById("titles");
    var languagesSelect = document.getElementById("languages");
    var note = document.getElementById("validation-note");

    if (!form || !titlesSelect || !languagesSelect) {
      return;
    }

    initializeMultiselect(titlesSelect, "Select titles");
    initializeMultiselect(languagesSelect, "Select languages");

    var refreshCounters = function () {
      updateCounter(titlesSelect, "titles-count");
      updateCounter(languagesSelect, "languages-count");
    };

    refreshCounters();

    titlesSelect.addEventListener("change", refreshCounters);
    languagesSelect.addEventListener("change", refreshCounters);

    form.addEventListener("submit", function (event) {
      var selectedTitleCount = countSelections(titlesSelect);
      if (selectedTitleCount < 1) {
        event.preventDefault();
        if (note) {
          note.textContent = "Please select at least 1 title before continuing.";
        }
        return;
      }

      if (note) {
        note.textContent = "";
      }
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    setStaggerFallbacks();
    initializePreferenceForm();
    document.body.classList.add("is-ready");
  });
})();