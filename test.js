$(document).ready(function () {
  let click_up = () => {
    if (
      !$("#number-of-games__0").val() ||
      parseInt($("#number-of-games__0").val()) < 50
    ) {
      var $button = $('button[aria-label="increase number of games"]');
      $button.click();
      setTimeout(() => {
        click_up();
      }, 0);
    }
  };

  let percent_down = () => {
    if (!$("#stake-on-win__0").val().includes("-")) {
      var $button = $('button[aria-label="decrease stop on profit value"]');
      $button.click();
    }
    if (!$("#stake-on-loss__0").val().includes("-")) {
      var $button = $('button[aria-label="decrease stop on loss value"]');
      $button.click();
    }
  };

  let profit_up = () => {
    if (parseFloat($("#stop-on__0").val()) < 0.001) {
      var $button = $('button[aria-label="Increase stop on profit value"]');
      $button.click();
      setTimeout(() => {
        profit_up();
      }, 0);
    }
  };

  let normalize = () => {
    click_up();
    percent_down();
    profit_up();
  };

  // normalize();

  let click_down = () => {
    if ($("#number-of-games__0").val() != "") {
      var $button = $('button[aria-label="decrease number of games"]');
      $button.click();
      setTimeout(() => {
        click_down();
      }, 0);
    }
  };

  let profit_down = () => {
    if (parseFloat($("#stop-on__0").val()) > 2e-8) {
      var $button = $('button[aria-label="Decrease stop on profit value"]');
      $button.click();
      setTimeout(() => {
        profit_down();
      }, 0);
    }
  };

  let percent_up = () => {
    if ($("#stake-on-win__0").val().includes("-")) {
      var $button = $('button[aria-label="increase stop on win value"]');
      $button.click();
    }
    if ($("#stake-on-loss__0").val().includes("-")) {
      var $button = $('button[aria-label="increase stop on loss value"]');
      $button.click();
    }
  };

  let denormalize = () => {
    click_down();
    percent_up();
    profit_down();
  };

  // denormalize();
  var anker = 0;

  setInterval(function () {
    let textPlay = $(
      "#root > div > div > div.sc-gZDVPe.kRyRgW > div > div > div:nth-child(1) > div.sc-elZpNk.iXiccI > div > div > div.sc-fdvtgc.hRINhW > div > div > div.sc-jfLonk.cWyHdI > button > div.sc-jZTQcj.bNjyff > div.sc-ejHFSJ.fSIGpu > div.sc-ZpFXC.ccpDSC"
    ).text();

    let bid = parseFloat(
      $(
        "#root > div > div > div.sc-gZDVPe.kRyRgW > div > div > div:nth-child(1) > div.sc-elZpNk.iXiccI > div > div > div.sc-fdvtgc.hRINhW > div > div > div:nth-child(3) > div > div > div:nth-child(2) > div > div.sc-csdhLT.eipjwl > input"
      ).val()
    );
    if (!$("#stop-on__0").val() && textPlay == "PLAY" && bid > 11e-8) {
      setTimeout(function () {
        let bal = $(
          "#root > div > div > div.sc-gZDVPe.kRyRgW > header > div > div.sc-UBoew.fBAzUq > div.sc-edctFj.dAxoqN > div.sc-fJVbdy.hIwmvz > span > span > span"
        ).text();
        if (anker == 0) anker = parseFloat(bal);
        if (bal >= anker + bid) {
          anker = bal - 2 * bid;
          // open from
          $(
            "#root > div > div > div.sc-gZDVPe.kRyRgW > div > div > div:nth-child(1) > div.sc-elZpNk.iXiccI > div > div > div.sc-fdvtgc.hRINhW > div > div > div.sc-jfLonk.cWyHdI > button > div.sc-jZTQcj.bNjyff > div.sc-iQmTax.abxyj"
          ).click();

          setTimeout(function () {
            normalize();
            setTimeout(function () {
              $(
                "#modals > div > div > div.sc-gpDAgj.iAwnVH > div > div > form > button"
              ).click();
            }, 3000);
          }, 1000);
        } else {
          $(
            "#root > div > div > div.sc-gZDVPe.kRyRgW > div > div > div:nth-child(1) > div.sc-elZpNk.iXiccI > div > div > div.sc-fdvtgc.hRINhW > div > div > div.sc-jfLonk.cWyHdI > button > div.sc-jZTQcj.bNjyff > div.sc-iQmTax.abxyj"
          ).click();

          setTimeout(function () {
            denormalize();
            setTimeout(function () {
              $(
                "#modals > div > div > div.sc-gpDAgj.iAwnVH > div > div > form > button"
              ).click();
            }, 3000);
          }, 1000);
        }
      }, 500);
    }
  }, 4000);

  setInterval(function () {
    let text = $(
      "#root > div > div > div.sc-hgaCvi.bQnfGO > div > div.sc-kIOfit.dqrGJe > div.sc-kqXUvJ.fztIEC > div > div"
    ).text();
    if (!text.includes("Next giveaway")) {
      $('[aria-label="Join giveaway"]').click();
    }
  }, 6000);
});
