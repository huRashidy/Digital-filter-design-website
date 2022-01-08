//window.onload = function() {draw()}
var unitCircleRadius = 200,
  zeroSize = 4,
  poleSize = 5,
  zeros = [],
  poles = [],
  canvas = document.getElementById('unitCircle')
  centerX = canvas.width / 2,
  centerY = canvas.height / 2,
  drag = false,
  del = false,
  show = true;
  let startX;
  let startY;
  lineChart = 0,
  lineChart2 = 0,
  lineChart3 = 0,
  lineChart4 = 0,
  lineChart5 = 0,
  lambda = 0,
  signalIterator=0,
  slidervalue =100,
  xData=[],
  yData=[],
  filtered=[];
for (let i = -0.9; i <= 0.9; i += 0.1) {
  if (i.toFixed(1) != -0.0) {
    document.getElementById("checkBoxes").innerHTML += "<input onclick  = \"freqResponse(" + i.toFixed(1) + ")\"  type=\"checkbox\" id=" + i.toFixed(1) + ">" + i.toFixed(1) + "<br>"
  } else {
    document.getElementById("checkBoxes").innerHTML += "<input onclick  = \"freqResponse(" + 0 + ")\" type=\"checkbox\" id=" + 0 + ">" + 0 + " <br> "
  }
}


$("#unitCircle").click(function (event) {
  getPosition(event);
});

canvas.addEventListener("contextmenu", e => e.preventDefault());
canvas.onmousedown = myDown;
canvas.onmouseup = myUp;
canvas.onmousemove = myMove;

var inRange = function (num, start, end) {
  if (!end) {
    end = start;
    start = 0;
  }
  return num >= start && num <= end;
};

function iterate(arr, range, x, y) {
  for (let i = 0; i < arr.length; i++) {
    xmax = arr[i].x + range
    xmin = arr[i].x - range
    ymax = arr[i].y + range
    ymin = arr[i].y - range
    if (inRange(x, xmin, xmax)) {
      if (inRange(y, ymin, ymax)) {
        return {
          flag: true,
          i
        }
      }
    }
  }
  return {
    flag: false,
    i: -1
  }
}

function getXY(event) {
  rect = canvas.getBoundingClientRect();
  return {
    x: event.clientX - rect.left,
    y: event.clientY - rect.top
  };
}

function draw() {
  var ctx = canvas.getContext('2d');
  var startAngle = 0;
  var endAngle = 2 * (Math.PI);

  ctx.beginPath();

  ctx.arc(centerX, centerY, unitCircleRadius, startAngle, endAngle);
  ctx.moveTo(centerX, centerY - unitCircleRadius);
  ctx.lineTo(centerX, centerY + unitCircleRadius);

  ctx.moveTo(centerX - unitCircleRadius, centerY);
  ctx.lineTo(centerX + unitCircleRadius, centerY);

  ctx.fillStyle = "black";
  ctx.fill();

  ctx.lineWidth = 4;
  ctx.strokeStyle = "#8a2f50";
  ctx.stroke();

  ctx.closePath();

  for (let i = 0; i < poles.length; i += 1) {
    var ctx = canvas.getContext("2d");
    ctx.beginPath();

    ctx.moveTo(poles[i].x - poleSize, poles[i].y - poleSize);
    ctx.lineTo(poles[i].x + poleSize, poles[i].y + poleSize);

    ctx.moveTo(poles[i].x + poleSize, poles[i].y - poleSize);
    ctx.lineTo(poles[i].x - poleSize, poles[i].y + poleSize);

    ctx.strokeStyle = "#ffffff";
    ctx.stroke();

    ctx.closePath();
  }

  for (let i = 0; i < zeros.length; i += 1) {
    var ctx = canvas.getContext("2d");
    ctx.beginPath();

    ctx.fillStyle = "#0000ff"; // Red color

    ctx.arc(zeros[i].x, zeros[i].y, zeroSize, 0, Math.PI * 2, true);

    ctx.fill();
    ctx.closePath();
  }
}



function getPosition(event) {
  var {
    x,
    y
  } = getXY(event)
  var flagZero = document.getElementById('zero').checked
  if (Math.pow((x - centerX), 2) + Math.pow((y - centerY), 2) <= Math.pow(unitCircleRadius, 2))
    drawCoordinates(x, y, flagZero);
}

function drawCoordinates(x, y, flagZero) {
  var ctx = canvas.getContext("2d");

  let exist = false
  let conj = document.getElementById('conj').checked

  if (iterate(zeros, zeroSize * 2, x, y).flag)
    exist = true
  if (iterate(poles, 10, x, y).flag)
    exist = true
  ctx.beginPath();

  if (!exist) {
    if (flagZero) {
      zeros.push({
        x,
        y,
        isDragging: false
      });
      ctx.fillStyle = "#ffffff"; // Red color

      ctx.arc(x, y, zeroSize, 0, Math.PI * 2, true);

      ctx.fill();
      if (conj) {
        y = 2 * centerY - y
        zeros.push({
          x,
          y,
          isDragging: false
        });
        ctx.fillStyle = "#ffffff"; // Red color

        ctx.arc(x, y, zeroSize, 0, Math.PI * 2, true);

        ctx.fill();

      }
    } else {
      poles.push({
        x,
        y,
        isDragging: false
      });
      ctx.moveTo(x - poleSize, y - poleSize);
      ctx.lineTo(x + poleSize, y + poleSize);

      ctx.moveTo(x + poleSize, y - poleSize);
      ctx.lineTo(x - poleSize, y + poleSize);

      ctx.strokeStyle = "#ffffff";
      ctx.stroke();
      if (conj) {
        y = 2 * centerY - y
        poles.push({
          x,
          y,
          isDragging: false
        });
        ctx.moveTo(x - poleSize, y - poleSize);
        ctx.lineTo(x + poleSize, y + poleSize);
        ctx.moveTo(x + poleSize, y - poleSize);
        ctx.lineTo(x - poleSize, y + poleSize);

        ctx.strokeStyle = "#ffffff";
        ctx.stroke();
      }
    }
  }

  ctx.closePath();
  freqResponse(5);
}

function clearCanvas(toClear) {
  if (toClear === 'zeros') {
    zeros = []
  } else if (toClear === 'poles') {
    poles = []
  } else {
    zeros = []
    poles = []
  }
  draw();
  freqResponse(5);
}

function myDown(event) {
  event.preventDefault();
  event.stopPropagation();

  var {
    x,
    y
  } = getXY(event)
  var rightClick = event.button == 2

  drag = false;
  del = false;

  var {
    flag,
    i
  } = iterate(zeros, zeroSize, x, y)

  if (flag) {
    if (rightClick) {
      del = true
      zeros.splice(i, 1)
    } else {
      drag = true;
      zeros[i].isDragging = true;
    }
  }

  var {
    flag,
    i
  } = iterate(poles, poleSize, x, y)
  if (!drag)
    if (flag) {
      if (rightClick) {
        del = true
        poles.splice(i, 1)
      } else {
        drag = true;
        poles[i].isDragging = true;
      }
    }
  startX = x;
  startY = y;
}

function myUp(event) {
  event.preventDefault();
  event.stopPropagation();

  if (!del) {
    drag = false
    for (var i = 0; i < zeros.length; i++) {
      zeros[i].isDragging = false;
    }
    for (var i = 0; i < poles.length; i++) {
      poles[i].isDragging = false;
    }

  } else {
    draw()
    freqResponse(5)
  }
}

function myMove(event) {
  if (drag) {
    event.preventDefault();
    event.stopPropagation();

    var {
      x,
      y
    } = getXY(event)

    var dx = x - startX;
    var dy = y - startY;

    for (var i = 0; i < zeros.length; i++) {
      if (zeros[i].isDragging) {
        zeros[i].x += dx;
        zeros[i].y += dy;
      }
    }

    for (var i = 0; i < poles.length; i++) {
      if (poles[i].isDragging) {
        poles[i].x += dx;
        poles[i].y += dy;
      }
    }

    draw();

    startX = x;
    startY = y;
  }
}

function showCheckboxes() {
  var checkboxes =
    document.getElementById("checkBoxes");

  if (show) {
    checkboxes.style.display = "block";
    show = false;
  } else {
    checkboxes.style.display = "none";
    show = true;
  }
}

function addFilter() {
  let filter = document.getElementById("text").value
  if (filter == "") return
  document.getElementById("checkBoxes").insertAdjacentHTML('beforeend', "<input onclick  = \"freqResponse('" + filter + "')\"  type=\"checkbox\" id=\"" + filter + "\">" + filter + "<br>")
}


function drawsignals(){
  const chart4 = document.getElementById("chart4")
  if (lineChart4 != 0)
        lineChart4.destroy();
  lineChart4 = new Chart(chart4, {
    type: 'line',
    data: {
      labels: xData,
      datasets: [{
        label: "ECG",
        fill: false,
        backgroundColor: "rgba(0,0,255,0.4)",
        borderColor: "rgba(0,0,255,0.7)",
        borderCapStyle: 'butt',
        borderDash: [],
        borderDashOffset: 0.0,
        borderJoinStyle: 'bevel',
        pointBorderColor: "rgba(0,0,255,1)",
        pointBackgroundColor: "#fff",
        pointBorderWidth: 1,
        pointHoverRadius: 5,
        pointHitRadius: 10,
        data: yData,

      },]
    }
  })
  const chart5 = document.getElementById("chart5")
  if (lineChart5 != 0)
        lineChart5.destroy();
  lineChart5 = new Chart(chart5, {
    type: 'line',
    data: {
      labels: xData,
      datasets: [{
        label: "ECG",
        fill: false,
        backgroundColor: "rgba(0,0,255,0.4)",
        borderColor: "rgba(0,0,255,0.7)",
        borderCapStyle: 'butt',
        borderDash: [],
        borderDashOffset: 0.0,
        borderJoinStyle: 'bevel',
        pointBorderColor: "rgba(0,0,255,1)",
        pointBackgroundColor: "#fff",
        pointBorderWidth: 1,
        pointHoverRadius: 5,
        pointHitRadius: 10,
        data: filtered,

      },]
    }
  })
}

function getSignals(){
arr = [signalIterator,slidervalue];
data = JSON.stringify(arr);
$.ajax({
url: '/reqsig',
type : 'post',
contentType: 'application/json',
dataType : 'json',
data :data,
success:function(response){
xData = xData.concat(response.xAxisData);
yData = yData.concat(response.yAxisData);
filtered = filtered.concat(response.filter);
length = response.datalength;
signalIterator = signalIterator + 1;
 drawsignals();
if(signalIterator*slidervalue < length){
  setTimeout(getSignals, 1000);
}
}
});
}

function freqResponse(lambda) {
  var zerosP = [],
    polesP = [];
  for (var i = 0; i < zeros.length; i++) {
    var xx = zeros[i].x - 230;
    var yy = 210 - zeros[i].y;
    xx /= 200.0;
    yy /= 200.0;
    zerosP.push([xx, yy])

  }
  for (var i = 0; i < poles.length; i++) {
    var xx = poles[i].x - 230;
    var yy = 210 - poles[i].y;
    xx /= 200.0;
    yy /= 200.0;
    polesP.push([xx, yy])
  }
  var flag = false;
  if (lambda != 5) {
    flag = document.getElementById(lambda).checked;
  }
  $.post("/postmethod", {
    zeros_data: JSON.stringify(zerosP),
    poles_data: JSON.stringify(polesP),
    lambdaP: JSON.stringify(lambda),
    flag: JSON.stringify(flag)

  },
    function (err, req, resp) {
      x = JSON.parse(resp["responseText"])
      if (lineChart != 0)
        lineChart.destroy();
      for (var i = 0; i < x.magnitudeX.length; i++) {
        x.magnitudeY[i] = x.magnitudeY[i].toFixed(2);
        x.angles[i] = x.angles[i].toFixed(2);
      }
    
    
      const chart = document.getElementById("chart")
      lineChart = new Chart(chart, {
        type: 'line',
        data: {
          labels: x.magnitudeX,
          datasets: [{
            label: "Magnitude Response",
            fill: false,
            backgroundColor: "rgba(75, 192, 192, 0.4)",
            borderColor: "rgba(75, 192, 192, 1)",
            borderCapStyle: 'butt',
            borderDash: [],
            borderDashOffset: 0.0,
            borderJoinStyle: 'miter',
            pointBorderColor: "rgba(75,192,192,1)",
            pointBackgroundColor: "#fff",
            pointBorderWidth: 1,
            pointHoverRadius: 5,
            pointHitRadius: 10,
            data: x.magnitudeY,
          },]
        }
      })
      if (lineChart2 != 0)
        lineChart2.destroy();

      const chart2 = document.getElementById("chart2")
      lineChart2 = new Chart(chart2, {
        type: 'line',
        data: {
          labels: x.magnitudeX,
          datasets: [{
            label: "Phase response",
            fill: false,
            backgroundColor: "rgba(255,140,0,0.4)",
            borderColor: "rgba(255,140,0,0.7)",
            borderCapStyle: 'butt',
            borderDash: [],
            borderDashOffset: 0.0,
            borderJoinStyle: 'miter',
            pointBorderColor: "rgba(255,140,0,1)",
            pointBackgroundColor: "#fff",
            pointBorderWidth: 1,
            pointHoverRadius: 5,
            pointHitRadius: 10,
            data: x.angles3,
          },]
        }
      })

      if (lineChart3 != 0)
        lineChart3.destroy();

      const chart3 = document.getElementById("chart3")
      lineChart3 = new Chart(chart3, {
        type: 'line',
        data: {
          labels: x.magnitudeX,
          datasets: [{
            label: "All-Pass Filters phase response",
            fill: false,
            backgroundColor: "rgba(34,139,34,0.4)",
            borderColor: "rgba(34,139,34,0.7)",
            borderCapStyle: 'butt',
            borderDash: [],
            borderDashOffset: 0.0,
            borderJoinStyle: 'bevel',
            pointBorderColor: "rgba(34,139,34,1)",
            pointBackgroundColor: "#fff",
            pointBorderWidth: 1,
            pointHoverRadius: 5,
            pointHitRadius: 10,
            data: x.angles2,
          },]
        }
      })
    });
  



(function () {
  var wf = document.createElement('script');
  wf.src = ('https:' == document.location.protocol ? 'https' : 'http') +
    '://ajax.googleapis.com/ajax/libs/webfont/1/webfont.js';
  wf.type = 'text/javascript';
  wf.async = 'true';
  var s = document.getElementsByTagName('script')[0];
  s.parentNode.insertBefore(wf, s);
})
();
}

