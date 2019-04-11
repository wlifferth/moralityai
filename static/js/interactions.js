var textContainer, textareaSize, input;
var autoSize = function () {
    // also can use textContent or innerText
    textareaSize.innerHTML = input.value + '\n';
};

document.addEventListener('DOMContentLoaded', function() {
    textContainer = document.querySelector('.textarea-container');
    textareaSize = textContainer.querySelector('.textarea-size');
    input = textContainer.querySelector('textarea');

    autoSize();
    input.addEventListener('input', autoSize);
});

document.addEventListener('DOMContentLoaded', function() {
    // Get the modal
    var successModal = document.getElementById('feedback-success-modal');
    var successModalX = document.getElementsByClassName("modal-close")[0];


    $("#positive-feedback").bind('click', function() {
        $.getJSON($SCRIPT_ROOT + '/_prediction_feedback', {
            sample_id: $('input[name="sample_id"]').val(),
            sample_text: $('input[name="prev_sample_text"]').val(),
            sample_prediction: $('input[name="prev_sample_prediction"]').val(),
            correct: true
        }, function(data) {
            successModal.style.display = "block";
            console.log(data)
        });
        return false;
    });

    $("#negative-feedback").bind('click', function() {
        $.getJSON($SCRIPT_ROOT + '/_prediction_feedback', {
            sample_id: $('input[name="sample_id"]').val(),
            sample_text: $('input[name="prev_sample_text"]').val(),
            sample_prediction: $('input[name="prev_sample_prediction"]').val(),
            correct: false
        }, function(data) {
            successModal.style.display = "block";
            console.log(data)
        });
        return false;
    });

    $('#main-text-input').keydown(function(event) {
        if (event.which == 13) {
            this.form.submit();
            event.preventDefault();
        }
    });

    successModalX.onclick = function() {
        successModal.style.display = "none";
    }

    window.onclick = function(event) {
        if (event.target == successModal) {
            successModal.style.display = "none";
        }
    }
});
