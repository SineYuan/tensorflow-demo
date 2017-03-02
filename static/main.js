var width  = 28
var height = 28


var customBoard = new DrawingBoard.Board('custom-board', {
    background: "#fff",
    color: "#000",
    size: 30,
    controls: [
        {Navigation: {back: false, forward: false}},
        {DrawingMode: {filler: false}}
    ],
    controlsPosition: "bottom right",
    webStorage: 'session',
    droppable: false
});

customBoard.ev.bind('board:stopDrawing', updateTinyBoard)
customBoard.ev.bind('board:reset', updateTinyBoard)

var goodStart = false
var tinyCan = $("#tiny")[0];
var tinyCtx = tinyCan.getContext("2d");
tinyCtx.scale(0.1, 0.1);

function updateTinyBoard() {
    if (true) {
        var imageData = customBoard.ctx.getImageData(0, 0, customBoard.canvas.width, customBoard.canvas.height);

        var newCanvas = $("<canvas>")
            .attr("width", imageData.width)
            .attr("height", imageData.height)[0];

        newCanvas.getContext("2d").putImageData(imageData, 0, 0);
        tinyCtx.drawImage(newCanvas, 0, 0);

    }
};

function convertURIToImageData(URI) {
    return new Promise(function (resolve, reject) {
        if (URI == null) return reject();
        var canvas = document.createElement('canvas'),
            context = canvas.getContext('2d'),
            image = new Image();
        image.addEventListener('load', function () {
            canvas.width = image.width;
            canvas.height = image.height;
            context.drawImage(image, 0, 0, canvas.width, canvas.height);
            resolve(context.getImageData(0, 0, canvas.width, canvas.height));
        }, false);
        image.src = URI;
    });
}

$('#Go').click(function(e) {
    console.log(e)
    var img = tinyCtx.getImageData(0, 0, width, height)
    var data = []

    // Loop through data.
    for (var i = 0; i < img.data.length; i += 4) {
        var myRed = img.data[i]; // First bytes are red bytes.
        var myGreen = img.data[i + 1]; // Second bytes are green bytes.
        var myBlue = img.data[i + 2]; // Third bytes are blue bytes.
        // Fourth bytes are alpha bytes

        // Assign average to red, green, and blue.
        myGray = parseInt((myRed + myGreen + myBlue) / 3); // Make it an integer.
        data.push(myGray)
    }
    var base64String = btoa(String.fromCharCode.apply(null, new Uint8Array(data)))
    var bytesArray = new Uint8Array(data)
    $.ajax({
       url: 'api/mnist',
       type: 'POST',
       contentType: 'application/octet-stream',
       data: bytesArray,
       processData: false,
       success: (data) => {
           console.log(data)
           if (data.Success) {
               var max = 0;
               var max_index = 0;
               for (let i = 0; i < 10; i++) {
                   var value = data.Result[i].toFixed(5);
                   if (value > max) {
                       max = value;
                       max_index = i;
                   }
                   $('#output tr').eq(i+1).find('td').eq(1).text(data.Result[i].toFixed(5));
               }
               console.log(max, max_index)

               for (let j = 0; j < 10; j++) {
                   if (j === max_index) {
                       $('#output tr').eq(j + 1).find('td').eq(1).addClass('success');
                   } else {
                       $('#output tr').eq(j + 1).find('td').eq(1).removeClass('success');
                   }
               }
           } else {
               alert(data.Msg)
           }
       }
    })
})
