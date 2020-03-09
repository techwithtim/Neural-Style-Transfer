// set correct active link in navbar
$(document).ready(function() {
    $.each($('#navbarNav').find('li'), function() {
        $(this).toggleClass('active', 
            window.location.pathname.indexOf($(this).find('a').attr('href')) > -1);
    }); 
});

$('#content-btn').click(function(){ 
    $('#content-upload').trigger('click'); 
    $("#content-upload").onchange = function() {
        $("img-form").submit();
    };
});
$('#style-btn').click(function(){ 
    $('#style-upload').trigger('click'); 
    $("#style-upload").onchange = function() {
        $("img-form").submit();
    };
});

async function get_default_image(){
    return await fetch('/get_default_image')
         .then(async function (response) {
            return await response.json();
        }).then(function (text) {
            return text["image-name"]
        });
  };

async function flash(message, color){
    // pass R or G for color
    await fetch('/flash_message?message=' + message + "&color=" + color);
}

async function generate(){
    var default_image = await get_default_image();
    var content_image = document.getElementById("content-img").src;
    var style_image = document.getElementById("style-img").src;
    //var output_image = document.getElementById("content-img").src;
    var check = [content_image, style_image];

    var valid = true;

    for (const c of check){
        let c_split = c.split("/");
        let path = c_split.slice(c_split.length-2, c_split.length);
        path = path[0] + "/" + path[1];
        if (path == default_image){
            valid=false;
        }
    }
    // submit form
    if(valid){
        document.getElementById("settings").submit();
    }else{
        await flash("Please upload a content and style image.", "R");
        location.reload();
    }

} 
