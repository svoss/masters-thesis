from email.MIMEMultipart import MIMEMultipart
from email.MIMEText import MIMEText
from email.MIMEImage import MIMEImage
import smtplib
import matplotlib as mpl
from slacker import Slacker

mpl.use('Agg')
import matplotlib.pyplot as plt


class Communication(object):
    def __init__(self, plot_folder):
        self.images = []
        self.text = []
        self.plot_folder = plot_folder
        self.files = []

    def add_plot(self, name, data):
        plt.plot(data)
        fig_name = name + ".png"
        full_path = self.plot_folder + "/" + fig_name
        plt.savefig(full_path, bbox_inches='tight')
        plt.close()
        self.add_image(full_path, fig_name)

    # Files are only send to slack!!
    def add_file(self, f, name):
        self.files.append((f, name))

    def add_image(self, f, name):
        self.images.append((f, name))

    def add_text(self, title, value):
        self.text.append((title, value))

    def get_text(self):
        return "\n".join([str(t) + ": " + str(v) for t, v in self.text])

    def send_slack(self, channel, api_token):
        # sends to slack
        slack = Slacker(api_token)
        text = self.get_text()
        for f, name in self.images:

            slack.files.upload(f, channels=channel, title=name)
        for f, name in self.files:
            slack.files.upload(f, channels=channel, title=name)
        if len(text) > 0:
            slack.chat.post_message(channel=channel, text=text, username='logbot', as_user='logbot')

    def send_mail(self, sender, receiver):
        # send mail
        msg = MIMEMultipart()

        # append text
        msg.attach(MIMEText(self.get_text()))

        for f, name in self.images:
            msg.attach(MIMEImage(file(f).read(), name=name))

        mailer = smtplib.SMTP()
        mailer.connect()
        mailer.sendmail(sender, receiver, msg.as_string())
        mailer.close()