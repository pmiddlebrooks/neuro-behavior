function slack_code_done(message)

if nargin == 0
    message = 'Code Done!';
end

    % webhook = 'https://hooks.slack.com/services/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';
    webhook = 'https://hooks.slack.com/services/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';
    
    % - Send the notification, with the attached message
    SendSlackNotification(webhook, message);
